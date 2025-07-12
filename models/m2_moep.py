#!/usr/bin/env python3
"""
M²-MOEP: Multi-scale Multi-expert Orthogonal Embedding Predictor
基于FFT+ms-Mamba的多尺度多专家时序预测模型
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .expert import FFTmsMambaExpert
from .flow import SimpleStableFlow as FlowModel


class M2_MOEP(nn.Module):
    """
    M²-MOEP主模型
    
    架构组成：
    1. 输入嵌入层 (Input Embedding)
    2. 位置编码 (Positional Encoding)
    3. 多专家网络 (Multi-Expert Network with FFT+ms-Mamba)
    4. 专家路由器 (Expert Router)
    5. 温度调度器 (Temperature Scheduler)
    6. 多尺度特征融合 (Multi-scale Feature Fusion)
    7. 预测头 (Prediction Head)
    """
    
    def __init__(self, config: Dict):
        super(M2_MOEP, self).__init__()
        
        self.config = config
        self.model_config = config['model']
        self.training_config = config.get('training', {})
        
        # 基础参数
        self.input_dim = self.model_config['input_dim']
        self.hidden_dim = self.model_config['hidden_dim']
        self.output_dim = self.model_config['output_dim']
        self.num_experts = self.model_config['num_experts']
        self.seq_len = self.model_config['seq_len']
        self.pred_len = self.model_config['pred_len']
        self.embedding_dim = self.model_config.get('embedding_dim', self.hidden_dim)
        
        # 设备管理
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device_initialized = False  # 添加设备初始化标志
        
        # 构建模型组件
        self._build_model()
        
        # 自动处理Mamba专家的设备
        self._handle_mamba_devices()
        
        # 初始化温度调度器
        self._init_temperature_scheduler()
        
        # 初始化损失统计
        self._init_loss_stats()
        
        # 模型参数初始化
        self._init_weights()
        
        print(f"✅ M²-MOEP模型初始化完成")
        print(f"   - 输入维度: {self.input_dim}")
        print(f"   - 隐藏维度: {self.hidden_dim}")
        print(f"   - 专家数量: {self.num_experts}")
        print(f"   - 总参数量: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_model(self):
        """构建模型架构 - 🔧 按照M²-MOEP文档重新设计"""
        
        # 1. 预训练的流式模型（核心组件）
        flow_config = self.model_config.get('flow', {})
        self.flow_model = FlowModel(
            input_dim=self.input_dim * self.seq_len,  # 扁平化输入
            flow_layers=flow_config.get('num_layers', 2)
        )
        
        # 2. 度量学习门控网络（孪生编码器）
        self.gating_encoder = nn.Sequential(
            nn.Linear(self.input_dim * self.seq_len, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 128)  # 嵌入向量维度
        )
        
        # 3. 可学习的专家原型（关键创新）
        self.expert_prototypes = nn.Parameter(
            torch.randn(self.num_experts, 128) * 0.01  # 小初始化
        )
        
        # 4. 专家网络（FFT+ms-Mamba）
        expert_configs = []
        for i in range(self.num_experts):
            expert_config = self.config.copy()
            expert_config['model'] = self.config['model'].copy()
            expert_config['model']['current_expert_id'] = i
            # 专家网络直接处理原始滑动窗口
            expert_config['model']['input_dim'] = self.input_dim
            expert_config['model']['output_dim'] = self.hidden_dim
            expert_configs.append(expert_config)
        
        self.experts = nn.ModuleList([
            FFTmsMambaExpert(expert_config) for expert_config in expert_configs
        ])
        
        # 5. 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 6. 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.pred_len * self.output_dim)
        )
        
        # 7. 温度参数（用于softmax路由）
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 8. 层归一化
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
        
        # 移除不必要的组件
        # - 删除位置编码（专家网络内部处理）
        # - 删除复杂的路由器（使用简单的原型距离）
        # - 删除多样性原型（专家原型已足够）
    
    def _handle_mamba_devices(self):
        """自动处理Mamba专家的设备切换"""
        cuda_available = torch.cuda.is_available()
        
        for i, expert in enumerate(self.experts):
            if hasattr(expert, 'use_mamba') and expert.use_mamba:
                if cuda_available:
                    print(f"专家{i}使用Mamba，自动切换到CUDA")
                    expert = expert.cuda()
                    # 更新专家列表中的引用
                    self.experts[i] = expert
                else:
                    print(f"专家{i}: CUDA不可用，Mamba专家将切换到LSTM模式")
                    expert.use_mamba = False
                    if hasattr(expert, '_init_lstm_fallback'):
                        expert._init_lstm_fallback()
    
    def _init_temperature_scheduler(self):
        """初始化温度调度器"""
        temp_config = self.model_config.get('temperature', {})
        initial_temp = temp_config.get('initial', 1.0)
        self.temperature.data = torch.tensor(initial_temp, dtype=torch.float32)
        
        self.temp_min = temp_config.get('min', 0.1)
        self.temp_max = temp_config.get('max', 5.0)
        self.temp_decay = temp_config.get('decay', 0.95)
        self.temp_schedule = temp_config.get('schedule', 'fixed')
        
        # 温度调度统计
        self.temp_stats = {
            'current': self.temperature.item(),
            'adjustments': 0,
            'performance_history': []
        }
    
    def _init_loss_stats(self):
        """初始化损失统计"""
        self.loss_stats = {
            'total_loss': 0.0,
            'prediction_loss': 0.0,
            'triplet_loss': 0.0,
            'diversity_loss': 0.0,
            'consistency_loss': 0.0,
            'reconstruction_loss': 0.0,
            'load_balance_loss': 0.0,
            'step_count': 0
        }
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                constant_(module.bias, 0)
                constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, ground_truth: torch.Tensor = None, 
                return_details: bool = False, return_aux_info: bool = False) -> Union[torch.Tensor, Dict]:
        """
        前向传播 - 🔧 严格按照M²-MOEP文档工作流程
        
        工作流程：
        1. 潜在空间映射：Wi → zi (通过Flow模型)
        2. 度量学习门控：zi → embi (通过孪生编码器)
        3. 专家路由：embi与专家原型计算距离 → αk
        4. 专家预测：原始输入gni → 专家输出
        5. 结果聚合：加权求和得到最终预测
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            ground_truth: 真实值张量，可选
            return_details: 是否返回详细信息
            return_aux_info: 是否返回辅助信息（兼容训练器）
            
        Returns:
            预测结果或详细信息字典
        """
        # return_aux_info 与 return_details 等价
        if return_aux_info:
            return_details = True
        
        batch_size, seq_len, input_dim = x.size()
        
        # 确保输入在正确设备上
        x = x.to(self.device)
        
        # 保存原始输入用于重构损失
        original_input = x.clone()
        
        # === 步骤1：潜在空间映射 Wi → zi ===
        # 扁平化输入用于Flow模型
        x_flat = x.view(batch_size, -1)  # [batch_size, seq_len * input_dim]
        
        try:
            # 通过Flow模型映射到潜在空间
            z_latent, flow_log_det = self.flow_model(x_flat)
            
            # 数值稳定性检查
            if torch.isnan(z_latent).any() or torch.isinf(z_latent).any():
                z_latent = x_flat
                flow_log_det = torch.zeros(batch_size, device=self.device)
                
        except Exception:
            # 简化异常处理
            z_latent = x_flat
            flow_log_det = torch.zeros(batch_size, device=self.device)
        
        # === 步骤2：度量学习门控 zi → embi ===
        # 通过孪生编码器生成嵌入向量
        embedding_vector = self.gating_encoder(z_latent)  # [batch_size, 128]
        
        # === 步骤3：专家路由 embi与专家原型计算距离 → αk ===
        # 计算嵌入向量与专家原型的距离
        distances = torch.cdist(
            embedding_vector.unsqueeze(1),  # [batch_size, 1, 128]
            self.expert_prototypes.unsqueeze(0)  # [1, num_experts, 128]
        ).squeeze(1)  # [batch_size, num_experts]
        
        # 使用负距离和温度参数计算路由权重
        routing_logits = -distances / torch.clamp(self.temperature, min=0.1, max=5.0)
        expert_weights = F.softmax(routing_logits, dim=-1)  # [batch_size, num_experts]
        
        # === 步骤4：专家预测 原始输入gni → 专家输出 ===
        expert_outputs = []
        expert_details = []
        
        # 优化：确保所有专家网络在同一设备上（避免重复检查）
        if not self._device_initialized:
            self._ensure_experts_on_device()
            self._device_initialized = True
        
        for i, expert in enumerate(self.experts):
            try:
                # 专家网络处理原始滑动窗口
                expert_output = expert(x, return_features=True)
                
                if isinstance(expert_output, dict):
                    expert_outputs.append(expert_output['output'])
                    expert_details.append(expert_output)
                else:
                    expert_outputs.append(expert_output)
                    expert_details.append({'output': expert_output})
                    
            except Exception:
                # 简化异常处理：使用零填充
                fallback_output = torch.zeros(batch_size, seq_len, self.hidden_dim, device=self.device)
                expert_outputs.append(fallback_output)
                expert_details.append({'output': fallback_output, 'error': True})
        
        # === 步骤5：结果聚合 加权求和得到最终预测 ===
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, seq_len, hidden_dim]
        expert_weights_expanded = expert_weights.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_experts, 1, 1]
        
        # 加权求和
        fused_output = torch.sum(expert_outputs * expert_weights_expanded, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # 特征融合
        fused_output = self.feature_fusion(fused_output)
        fused_output = self.layer_norm(fused_output)
        
        # 预测头：使用最后一个时间步的特征进行预测
        last_hidden = fused_output[:, -1, :]  # [batch_size, hidden_dim]
        predictions = self.prediction_head(last_hidden)  # [batch_size, pred_len * output_dim]
        predictions = predictions.view(batch_size, self.pred_len, self.output_dim)
        predictions = self.output_norm(predictions)
        
        if return_details:
            # 🔧 修复：正确生成expert_embeddings
            expert_embeddings = embedding_vector  # [batch_size, 128]
            
            # 构建完整的输出字典
            output_dict = {
                'predictions': predictions,
                'expert_weights': expert_weights,
                'expert_outputs': expert_outputs,
                'expert_details': expert_details,
                'fused_features': fused_output,
                'latent_features': z_latent,
                'embedding_vector': embedding_vector,
                'expert_prototypes': self.expert_prototypes,
                'routing_distances': distances,
                'routing_logits': routing_logits,
                'flow_log_det': flow_log_det,
                'hidden_states': last_hidden,
                'temperature': self.temperature.item(),
                'loss_stats': self.loss_stats,
                'original_input': original_input,  # 添加原始输入用于重构损失
                # 🔧 修复：添加正确的expert_embeddings到aux_info
                'aux_info': {
                    'expert_weights': expert_weights,
                    'expert_embeddings': expert_embeddings,  # 🔧 关键修复
                    'flow_embeddings': z_latent,
                    'flow_log_det': flow_log_det,
                    'routing_entropy': -torch.sum(expert_weights * torch.log(expert_weights + 1e-8), dim=-1).mean().item(),
                    'temperature': self.temperature.item(),
                    'num_experts_used': (expert_weights > 0.01).sum(dim=1).float().mean().item(),
                    'prototype_distances': distances.mean(dim=0).tolist(),
                    'original_input': original_input  # 添加原始输入
                }
            }
            
            return output_dict
        else:
            return {'predictions': predictions}
    
    def compute_loss(self, outputs: Dict, targets: torch.Tensor, epoch: int = 0) -> Dict:
        """
        计算M²-MOEP复合损失函数 - 🔧 按照文档要求重新实现
        
        复合损失包含：
        1. 重构损失 (Lrc): 确保Flow模型保留原始序列信息
        2. 路由损失 (Lcl): 三元组损失训练度量学习门控
        3. 预测损失 (Lpr): 主要的预测性能指标
        
        Args:
            outputs: 模型输出字典
            targets: 目标张量 [batch_size, pred_len, output_dim]
            epoch: 当前训练轮次
            
        Returns:
            损失字典
        """
        predictions = outputs['predictions']
        aux_info = outputs.get('aux_info', {})
        
        # 确保预测和目标在同一设备
        predictions = predictions.to(targets.device)
        
        # 损失权重配置
        loss_weights = self.training_config.get('loss_weights', {})
        
        # === 1. 预测损失 (Lpr) - 主要损失 ===
        prediction_loss = F.mse_loss(predictions, targets)
        
        # === 2. 重构损失 (Lrc) - Flow模型保真度 ===
        reconstruction_loss = self._compute_flow_reconstruction_loss(
            outputs.get('latent_features'),
            outputs.get('flow_log_det'),
            outputs.get('original_input')  # 修复：使用原始输入
        )
        
        # === 3. 路由损失 (Lcl) - 三元组损失 ===
        triplet_loss = self._compute_triplet_routing_loss(
            aux_info.get('expert_embeddings'),
            aux_info.get('expert_weights'),
            predictions,
            targets
        )
        
        # === 4. 专家原型正则化损失 ===
        prototype_reg_loss = self._compute_prototype_regularization()
        
        # === 5. 负载均衡损失 ===
        load_balance_loss = self._compute_load_balance_loss(
            aux_info.get('expert_weights')
        )
        
        # === 复合损失计算 ===
        # 使用文档中提到的不确定性加权方法（简化版）
        total_loss = (
            loss_weights.get('prediction', 1.0) * prediction_loss +
            loss_weights.get('reconstruction', 0.1) * reconstruction_loss +
            loss_weights.get('triplet', 0.1) * triplet_loss +
            loss_weights.get('prototype_reg', 0.01) * prototype_reg_loss +
            loss_weights.get('load_balance', 0.01) * load_balance_loss
        )
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("⚠️ 检测到NaN/Inf损失，使用备用损失")
            total_loss = prediction_loss
        
        # 更新损失统计
        self.loss_stats.update({
            'total_loss': total_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'prototype_reg_loss': prototype_reg_loss.item(),
            'load_balance_loss': load_balance_loss.item(),
            'step_count': self.loss_stats['step_count'] + 1
        })
        
        # 返回损失字典 - 统一键名格式
        return {
            'total': total_loss,
            'prediction': prediction_loss,
            'reconstruction': reconstruction_loss,
            'triplet': triplet_loss,
            'prototype': prototype_reg_loss,
            'load_balance': load_balance_loss
        }
    
    def _compute_flow_reconstruction_loss(self, latent_features: torch.Tensor, 
                                        flow_log_det: torch.Tensor, 
                                        original_input: torch.Tensor) -> torch.Tensor:
        """计算Flow模型重构损失 - 修复：使用原始输入作为参考"""
        if latent_features is None or original_input is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            # 重构原始输入
            batch_size = latent_features.size(0)
            reconstructed = self.flow_model.reconstruct(latent_features)
            
            # 将重构结果reshape回原始输入形状
            reconstructed_input = reconstructed.view(batch_size, self.seq_len, self.input_dim)
            
            # 使用原始输入作为重构参考
            reconstruction_mse = F.mse_loss(reconstructed_input, original_input)
            
            # 添加Flow模型的对数行列式项（正则化）
            if flow_log_det is not None:
                log_det_reg = torch.mean(flow_log_det ** 2) * 0.01
                reconstruction_loss = reconstruction_mse + log_det_reg
            else:
                reconstruction_loss = reconstruction_mse
            
            # 数值稳定性检查
            if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            return reconstruction_loss
            
        except Exception as e:
            print(f"⚠️ Flow重构损失计算失败: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _compute_triplet_routing_loss(self, expert_embeddings: torch.Tensor,
                                    expert_weights: torch.Tensor,
                                    predictions: torch.Tensor,
                                    targets: torch.Tensor) -> torch.Tensor:
        """计算三元组路由损失 - 按照文档要求实现"""
        if expert_embeddings is None or expert_weights is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            batch_size = expert_embeddings.size(0)
            if batch_size < 3:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 计算预测误差，用于构建三元组
            prediction_errors = F.mse_loss(predictions, targets, reduction='none')
            prediction_errors = prediction_errors.mean(dim=(1, 2))  # [batch_size]
            
            # 根据预测误差排序，构建三元组
            sorted_indices = torch.argsort(prediction_errors)
            
            # 选择锚点、正样本、负样本
            num_triplets = min(batch_size // 3, 10)  # 限制三元组数量
            if num_triplets == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            anchors = expert_embeddings[sorted_indices[:num_triplets]]
            positives = expert_embeddings[sorted_indices[num_triplets:2*num_triplets]]
            negatives = expert_embeddings[sorted_indices[-num_triplets:]]
            
            # 计算三元组损失
            pos_dist = F.pairwise_distance(anchors, positives, 2)
            neg_dist = F.pairwise_distance(anchors, negatives, 2)
            
            margin = self.model_config.get('triplet', {}).get('margin', 0.5)
            triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()
            
            # 数值稳定性检查
            if torch.isnan(triplet_loss) or torch.isinf(triplet_loss):
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            return triplet_loss
            
        except Exception as e:
            print(f"⚠️ 三元组损失计算失败: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _compute_prototype_regularization(self) -> torch.Tensor:
        """计算专家原型正则化损失"""
        try:
            # 鼓励专家原型之间的多样性
            prototypes = self.expert_prototypes  # [num_experts, 128]
            
            # 计算原型之间的相似度矩阵
            similarity_matrix = torch.mm(prototypes, prototypes.t())  # [num_experts, num_experts]
            
            # 除去对角线元素（自身相似度）
            mask = torch.eye(self.num_experts, device=self.device)
            off_diagonal = similarity_matrix * (1 - mask)
            
            # 最小化非对角线元素（鼓励原型多样性）
            diversity_loss = torch.mean(off_diagonal ** 2)
            
            # 防止原型范数过大
            norm_reg = torch.mean(torch.norm(prototypes, dim=1) ** 2) * 0.01
            
            return diversity_loss + norm_reg
            
        except Exception as e:
            print(f"⚠️ 原型正则化损失计算失败: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _compute_load_balance_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """计算专家负载均衡损失，鼓励所有专家被使用"""
        if expert_weights is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            # 计算专家权重的均匀性
            mean_weights = expert_weights.mean(dim=0)  # [num_experts]
            target_weight = 1.0 / self.num_experts
            
            # 鼓励权重分布均匀
            load_balance_loss = torch.mean((mean_weights - target_weight) ** 2)
            
            return load_balance_loss
            
        except Exception as e:
            print(f"⚠️ 负载均衡损失计算失败: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def update_temperature(self, performance_metric: float, epoch: int):
        """
        根据指定的策略更新温度参数

        Args:
            performance_metric (float): 用于决策的性能指标（例如验证损失）
            epoch (int): 当前训练轮次
        """
        if self.temp_schedule == 'fixed':
            return  # 固定温度，不更新

        # 记录性能，用于未来可能的自适应策略
        self.temp_stats['performance_history'].append(performance_metric)
        self.temp_stats['adjustments'] += 1

        if self.temp_schedule == 'exponential':
            # 指数衰减
            new_temp = self.temperature.item() * self.temp_decay
        elif self.temp_schedule == 'cosine':
            # 余弦退火
            total_epochs = self.training_config.get('epochs', 1)  # 避免除以零
            initial_temp = self.model_config.get('temperature', {}).get('initial', self.temp_max)
            
            cosine_val = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
            new_temp = self.temp_min + (initial_temp - self.temp_min) * cosine_val
        else:
            # 默认为指数衰减
            new_temp = self.temperature.item() * self.temp_decay

        # 限制温度在预设的范围内
        clamped_temp = max(self.temp_min, min(new_temp, self.temp_max))
        
        # 修复：确保温度参数在正确设备上
        self.temperature.data = torch.tensor(clamped_temp, dtype=torch.float32, device=self.temperature.device)
        self.temp_stats['current'] = self.temperature.item()

    def to(self, device):
        """重写to方法，确保所有组件都移动到正确设备"""
        super().to(device)
        
        # 确保所有专家网络都在正确设备上
        # 对于使用mamba的专家网络，需要特别处理
        for i, expert in enumerate(self.experts):
            try:
                expert.to(device)
            except Exception as e:
                print(f"⚠️ 专家{i}移动到设备{device}时出错: {e}")
                # 如果是mamba相关的错误，可能需要强制在CUDA上
                if hasattr(expert, 'use_mamba') and expert.use_mamba:
                    if device.type != 'cuda' and torch.cuda.is_available():
                        print(f"⚠️ 专家{i}使用mamba，强制移动到CUDA")
                        expert.to(torch.device('cuda'))
                    else:
                        print(f"⚠️ 专家{i}切换到LSTM模式")
                        expert.use_mamba = False
                        # 重新初始化为LSTM
                        expert._init_lstm_fallback()
                        expert.to(device)
                else:
                    expert.to(device)
        
        self.device = device
        return self
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        expert_params = []
        for i, expert in enumerate(self.experts):
            expert_param_count = sum(p.numel() for p in expert.parameters())
            expert_params.append({
                'expert_id': i,
                'parameters': expert_param_count,
                'use_mamba': getattr(expert, 'use_mamba', False)
            })
        
        return {
            'model_name': 'M²-MOEP',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'expert_parameters': expert_params,
            'num_experts': self.num_experts,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'temperature': self.temperature,
            'device': str(self.device)
        }

    def _ensure_experts_on_device(self):
        """确保所有专家网络都在正确设备上（只在初始化时调用一次）"""
        for i, expert in enumerate(self.experts):
            try:
                expert.to(self.device)
            except Exception as e:
                print(f"⚠️ 专家{i}移动到设备{self.device}时出错: {e}")
                # 如果是mamba相关的错误，可能需要特殊处理
                if hasattr(expert, 'use_mamba') and expert.use_mamba:
                    if self.device.type != 'cuda' and torch.cuda.is_available():
                        print(f"⚠️ 专家{i}使用mamba，强制移动到CUDA")
                        expert.to(torch.device('cuda'))
                    else:
                        print(f"⚠️ 专家{i}切换到LSTM模式")
                        expert.use_mamba = False
                        if hasattr(expert, '_init_lstm_fallback'):
                            expert._init_lstm_fallback()
                        expert.to(self.device)
                else:
                    expert.to(self.device) 