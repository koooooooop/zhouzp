import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from .expert import FFTmsMambaExpert
from .gating import GatingEncoder
from .flow import PowerfulNormalizingFlow


class M2_MOEP(nn.Module):
    """
    M²-MOEP: Mamba-Metric Mixture of Experts Predictor
    
    核心特性：
    - 预训练的Flow模型进行潜在表示映射
    - 基于度量学习的门控机制（接受潜在表示）
    - FFT+ms-Mamba专家网络（早期融合）
    - 基于预测性能的Triplet Loss
    - 端到端训练
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 基础配置
        self.config = config
        self.input_dim = config['model']['input_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.output_dim = config['model']['output_dim']
        self.num_experts = config['model']['num_experts']
        self.seq_len = config['model']['seq_len']
        self.pred_len = config['model']['pred_len']
        
        # Flow模型配置
        flow_config = config['model'].get('flow', {})
        self.flow_latent_dim = flow_config.get('latent_dim', 256)
        self.use_pretrained_flow = flow_config.get('use_pretrained', True)
        
        # 专家多样性配置
        self.diversity_config = config['model'].get('diversity', {})
        self.prototype_dim = self.diversity_config.get('prototype_dim', 64)
        self.num_prototypes = self.diversity_config.get('num_prototypes', self.num_experts * 2)
        self.diversity_weight = self.diversity_config.get('diversity_weight', 0.1)
        self.force_diversity = self.diversity_config.get('force_diversity', True)
        
        # 温度调度配置
        temp_config = config['model'].get('temperature', {})
        self.initial_temperature = temp_config.get('initial', 1.0)
        self.min_temperature = temp_config.get('min', 0.1)
        self.max_temperature = temp_config.get('max', 10.0)
        self.temperature_decay = temp_config.get('decay', 0.95)
        self.temperature_schedule = temp_config.get('schedule', 'exponential')
        
        # 可学习的log温度参数
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(self.initial_temperature)))
        
        # === 核心组件 ===
        # 1. Flow模型（用于潜在表示映射）
        flow_input_dim = self.seq_len * self.input_dim  # 展平后的输入维度
        self.flow_model = PowerfulNormalizingFlow(
            input_dim=flow_input_dim,
            latent_dim=self.flow_latent_dim,
            hidden_dim=self.hidden_dim,
            num_coupling_layers=6
        )
        
        # 2. 门控网络（接受潜在表示）
        # 更新配置以传递潜在维度
        gating_config = config.copy()
        gating_config['model']['latent_dim'] = self.flow_latent_dim
        self.gating = GatingEncoder(gating_config)
        
        # 3. 专家网络（FFT+ms-Mamba）
        self.experts = nn.ModuleList([
            FFTmsMambaExpert(config) for _ in range(self.num_experts)
        ])
        
        # === Triplet Loss 组件 ===
        # 三元组损失配置
        triplet_config = config['model'].get('triplet', {})
        self.triplet_margin = triplet_config.get('margin', 0.5)
        self.triplet_mining_strategy = triplet_config.get('mining_strategy', 'batch_hard')
        self.triplet_loss_weight = triplet_config.get('loss_weight', 1.0)
        
        # 专家性能追踪（用于三元组挖掘）
        self.expert_performance_history = {}
        self.performance_window_size = triplet_config.get('performance_window', 100)
        
        # 原型分离组件
        self.prototype_projector = nn.Linear(self.prototype_dim, self.prototype_dim)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        # 专家特征提取器（用于多样性计算）
        # 输入维度：7个特征类型 * input_dim
        expert_feature_input_dim = 7 * self.input_dim
        self.expert_feature_extractor = nn.Sequential(
            nn.Linear(expert_feature_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.prototype_dim),
            nn.LayerNorm(self.prototype_dim)
        )
        
        # 辅助损失权重
        self.aux_loss_weight = config['training'].get('aux_loss_weight', 0.01)
        
        # 训练状态
        self.training_step = 0
        self.current_epoch = 0
    
    @property
    def temperature(self):
        """动态温度属性，确保在合理范围内"""
        temp = torch.exp(self.log_temperature)
        return torch.clamp(temp, self.min_temperature, self.max_temperature)
    
    @temperature.setter
    def temperature(self, value):
        """设置温度值"""
        # 确保值在合理范围内
        value = max(self.min_temperature, min(self.max_temperature, value))
        # 更新log_temperature参数
        self.log_temperature.data = torch.log(torch.tensor(value))
    
    def update_temperature_schedule(self, epoch, expert_entropy):
        """
        改进的温度调度策略
        基于专家使用熵的自适应调整，更加稳健
        """
        # 计算归一化熵（0-1范围）
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=expert_entropy.device))
        normalized_entropy = expert_entropy / max_entropy
        
        # 基于epoch的基础温度衰减（更温和）
        epoch_factor = max(0.5, 1.0 - epoch * 0.008)  # 从1.0缓慢衰减到0.5
        
        # 基于熵的自适应因子
        if normalized_entropy < 0.4:  # 专家使用过于集中
            entropy_factor = 1.8  # 提高温度，增加探索
        elif normalized_entropy < 0.6:  # 轻微不均衡
            entropy_factor = 1.3
        elif normalized_entropy > 0.9:  # 过于分散，可能影响性能
            entropy_factor = 0.8  # 降低温度，增加确定性
        else:  # 理想范围 [0.6, 0.9]
            entropy_factor = 1.0  # 保持当前温度
        
        # 计算新温度
        new_temperature = self.initial_temperature * epoch_factor * entropy_factor
        new_temperature = torch.clamp(
            torch.tensor(new_temperature, device=self.log_temperature.device),
            self.min_temperature, 
            self.max_temperature
        )
        
        # 使用指数移动平均平滑温度变化
        current_temp = torch.exp(self.log_temperature)
        smoothed_temp = 0.9 * current_temp + 0.1 * new_temperature
        
        # 更新log_temperature参数
        self.log_temperature.data = torch.log(smoothed_temp)
    
    def compute_expert_features(self, x: torch.Tensor) -> torch.Tensor:
        """计算更丰富的专家特征用于多样性分析"""
        batch_size, seq_len, input_dim = x.shape
        
        # 1. 统计特征
        x_mean = x.mean(dim=1)        # [batch_size, input_dim]
        x_std = x.std(dim=1)          # [batch_size, input_dim]
        x_min = x.min(dim=1)[0]       # [batch_size, input_dim]
        x_max = x.max(dim=1)[0]       # [batch_size, input_dim]
        
        # 2. 时序特征
        # 计算一阶差分（变化率）
        x_diff = torch.diff(x, dim=1)  # [batch_size, seq_len-1, input_dim]
        x_diff_mean = x_diff.mean(dim=1)  # [batch_size, input_dim]
        
        # 计算趋势（线性回归斜率近似）
        time_steps = torch.arange(seq_len, device=x.device, dtype=x.dtype).view(1, -1, 1)
        time_steps = time_steps.expand(batch_size, seq_len, input_dim)
        
        # 简化的趋势计算：最后值与第一值的差除以时间长度
        x_trend = (x[:, -1, :] - x[:, 0, :]) / seq_len  # [batch_size, input_dim]
        
        # 3. 频域特征（简化）
        fft_x = torch.fft.fft(x, dim=1)
        x_magnitude_mean = torch.abs(fft_x).mean(dim=1)  # [batch_size, input_dim]
        
        # 4. 组合所有特征
        combined_features = torch.cat([
            x_mean, x_std, x_min, x_max,  # 统计特征
            x_diff_mean, x_trend,          # 时序特征
            x_magnitude_mean               # 频域特征
        ], dim=1)  # [batch_size, 7*input_dim]
        
        # 5. 降维投影
        expert_features = self.expert_feature_extractor(combined_features)
        
        return expert_features
    
    def compute_prototype_separation_loss(self, expert_features: torch.Tensor, 
                                        expert_weights: torch.Tensor) -> torch.Tensor:
        """
        改进的原型分离损失 - 使用对比学习策略
        目标：确保不同专家的原型在特征空间中分离
        """
        batch_size = expert_features.size(0)
        
        # 1. 投影到原型空间
        projected_features = self.prototype_projector(expert_features)  # [batch_size, prototype_dim]
        
        # 2. 计算样本与所有原型的相似度
        prototype_similarities = F.cosine_similarity(
            projected_features.unsqueeze(1),  # [batch_size, 1, prototype_dim]
            self.prototypes.unsqueeze(0),     # [1, num_prototypes, prototype_dim]
            dim=2
        )  # [batch_size, num_prototypes]
        
        # 3. 基于专家权重确定目标原型分配
        # 使用专家权重作为软目标
        target_expert = torch.argmax(expert_weights, dim=1)  # [batch_size]
        
        # 每个专家对应的原型范围
        prototypes_per_expert = self.num_prototypes // self.num_experts
        
        # 4. 计算原型分离损失
        separation_losses = []
        
        for expert_id in range(self.num_experts):
            # 当前专家的样本mask
            expert_mask = (target_expert == expert_id)
            if not expert_mask.any():
                continue
            
            # 当前专家对应的原型索引
            start_idx = expert_id * prototypes_per_expert
            end_idx = min(start_idx + prototypes_per_expert, self.num_prototypes)
            
            # 当前专家的样本
            expert_samples = projected_features[expert_mask]  # [num_samples, prototype_dim]
            expert_similarities = prototype_similarities[expert_mask]  # [num_samples, num_prototypes]
            
            if expert_samples.size(0) == 0:
                continue
            
            # 正样本：当前专家的原型
            positive_similarities = expert_similarities[:, start_idx:end_idx]  # [num_samples, prototypes_per_expert]
            positive_max = positive_similarities.max(dim=1)[0]  # [num_samples]
            
            # 负样本：其他专家的原型
            negative_mask = torch.ones(self.num_prototypes, dtype=torch.bool, device=expert_similarities.device)
            negative_mask[start_idx:end_idx] = False
            negative_similarities = expert_similarities[:, negative_mask]  # [num_samples, other_prototypes]
            negative_max = negative_similarities.max(dim=1)[0]  # [num_samples]
            
            # 对比损失：拉近正样本，推远负样本
            margin = 0.5
            contrastive_loss = F.relu(negative_max - positive_max + margin)
            separation_losses.append(contrastive_loss.mean())
        
        if not separation_losses:
            return torch.tensor(0.0, device=expert_features.device)
        
        # 5. 添加原型间的排斥损失
        # 确保不同专家的原型彼此远离
        prototype_distances = torch.cdist(self.prototypes, self.prototypes, p=2)  # [num_prototypes, num_prototypes]
        
        # 创建专家分组mask
        expert_groups = torch.arange(self.num_prototypes, device=self.prototypes.device) // prototypes_per_expert
        same_expert_mask = expert_groups.unsqueeze(0) == expert_groups.unsqueeze(1)
        different_expert_mask = ~same_expert_mask
        
        # 不同专家的原型应该距离较远
        if different_expert_mask.any():
            inter_expert_distances = prototype_distances[different_expert_mask]
            repulsion_loss = F.relu(1.0 - inter_expert_distances).mean()  # 距离小于1时施加惩罚
        else:
            repulsion_loss = torch.tensor(0.0, device=expert_features.device)
        
        # 组合损失
        total_separation_loss = torch.stack(separation_losses).mean() + 0.1 * repulsion_loss
        
        return total_separation_loss
    
    def inject_diversity_noise(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        改进的多样性噪声注入机制
        使用更智能的策略防止专家崩塌
        """
        if not self.force_diversity or not self.training:
            return expert_weights
        
        batch_size, num_experts = expert_weights.shape
        
        # 1. 计算专家使用分布
        expert_usage = expert_weights.mean(dim=0)  # [num_experts]
        usage_entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
        max_entropy = torch.log(torch.tensor(float(num_experts), device=expert_weights.device))
        normalized_entropy = usage_entropy / max_entropy
        
        # 2. 检测专家崩塌
        # 崩塌指标：最大使用率过高 + 熵过低
        max_usage = expert_usage.max()
        min_usage = expert_usage.min()
        usage_ratio = max_usage / (min_usage + 1e-8)
        
        # 3. 自适应噪声强度
        if normalized_entropy < 0.5 or usage_ratio > 5.0:
            # 严重不均衡：较强噪声
            noise_strength = 0.15
            noise_type = 'strong'
        elif normalized_entropy < 0.7 or usage_ratio > 3.0:
            # 轻微不均衡：中等噪声
            noise_strength = 0.08
            noise_type = 'medium'
        else:
            # 均衡状态：轻微噪声或无噪声
            noise_strength = 0.02
            noise_type = 'light'
        
        # 4. 智能噪声注入策略
        if noise_type == 'strong':
            # 强噪声：重新平衡专家权重
            # 对使用率低的专家给予额外的探索机会
            exploration_bonus = torch.zeros_like(expert_weights)
            
            # 给使用率低于平均值的专家加分
            below_avg_mask = expert_usage < expert_usage.mean()
            if below_avg_mask.any():
                exploration_bonus[:, below_avg_mask] = noise_strength
            
            # 添加小量随机噪声
            random_noise = torch.randn_like(expert_weights) * (noise_strength * 0.3)
            
            noisy_weights = expert_weights + exploration_bonus + random_noise
            
        elif noise_type == 'medium':
            # 中等噪声：温和的随机化
            adaptive_noise = torch.randn_like(expert_weights) * noise_strength
            # 对主导专家施加更多噪声
            dominant_expert = torch.argmax(expert_usage)
            adaptive_noise[:, dominant_expert] *= 1.5
            
            noisy_weights = expert_weights + adaptive_noise
            
        else:  # light noise
            # 轻噪声：最小扰动
            light_noise = torch.randn_like(expert_weights) * noise_strength
            noisy_weights = expert_weights + light_noise
        
        # 5. 重新归一化并应用温度
        noisy_weights = F.softmax(noisy_weights / self.temperature, dim=-1)
        
        # 6. 使用指数移动平均平滑变化（避免剧烈波动）
        if hasattr(self, '_prev_expert_weights'):
            alpha = 0.8 if noise_type == 'strong' else 0.9
            smoothed_weights = alpha * noisy_weights + (1 - alpha) * self._prev_expert_weights
        else:
            smoothed_weights = noisy_weights
        
        # 保存当前权重用于下次平滑
        self._prev_expert_weights = smoothed_weights.detach()
        
        return smoothed_weights
    
    def mine_triplets_based_on_prediction_performance(self, x: torch.Tensor, 
                                                     expert_weights: torch.Tensor,
                                                     expert_predictions: torch.Tensor,
                                                     ground_truth: torch.Tensor) -> List[Tuple[int, int, int]]:
        """
        基于预测性能挖掘三元组
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            expert_weights: 专家权重 [batch_size, num_experts]
            expert_predictions: 专家预测 [batch_size, num_experts, pred_len, input_dim]
            ground_truth: 真实值 [batch_size, pred_len, input_dim]
            
        Returns:
            三元组列表 [(anchor_idx, positive_idx, negative_idx), ...]
        """
        batch_size = x.size(0)
        triplets = []
        
        # 计算每个专家对每个样本的预测误差
        expert_errors = torch.zeros(batch_size, self.num_experts, device=x.device)
        for i in range(self.num_experts):
            # expert_predictions[:, i, :, :] 是 [batch_size, pred_len, input_dim]
            # ground_truth 是 [batch_size, pred_len, input_dim]
            expert_errors[:, i] = F.mse_loss(
                expert_predictions[:, i, :, :], 
                ground_truth, 
                reduction='none'
            ).mean(dim=(1, 2))  # 对pred_len和input_dim维度求平均
        
        # 为每个样本找到最佳专家
        best_experts = torch.argmin(expert_errors, dim=1)  # [batch_size]
        
        # 构建三元组
        for anchor_idx in range(batch_size):
            anchor_best_expert = best_experts[anchor_idx].item()
            
            # 寻找正样本：同样由该专家主导预测且预测准确的样本
            positive_candidates = []
            negative_candidates = []
            
            for candidate_idx in range(batch_size):
                if candidate_idx == anchor_idx:
                    continue
                    
                candidate_best_expert = best_experts[candidate_idx].item()
                
                if candidate_best_expert == anchor_best_expert:
                    # 同一专家主导，作为正样本候选
                    positive_candidates.append(candidate_idx)
                else:
                    # 不同专家主导，作为负样本候选
                    negative_candidates.append(candidate_idx)
            
            # 选择最佳正样本和负样本
            if positive_candidates and negative_candidates:
                # 选择预测误差最小的正样本
                positive_errors = expert_errors[positive_candidates, anchor_best_expert]
                best_positive_idx = positive_candidates[torch.argmin(positive_errors).item()]
                
                # 选择预测误差最大的负样本（最难区分的负样本）
                negative_errors = expert_errors[negative_candidates, anchor_best_expert]
                hardest_negative_idx = negative_candidates[torch.argmax(negative_errors).item()]
                
                triplets.append((anchor_idx, best_positive_idx, hardest_negative_idx))
        
        return triplets
    
    def compute_triplet_loss(self, embeddings: torch.Tensor, 
                           triplets: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        计算三元组损失
        
        Args:
            embeddings: 嵌入向量 [batch_size, embedding_dim]
            triplets: 三元组列表
            
        Returns:
            三元组损失
        """
        if not triplets:
            return torch.tensor(0.0, device=embeddings.device)
        
        triplet_losses = []
        
        for anchor_idx, positive_idx, negative_idx in triplets:
            anchor_emb = embeddings[anchor_idx]
            positive_emb = embeddings[positive_idx]
            negative_emb = embeddings[negative_idx]
            
            # 计算距离
            pos_dist = F.pairwise_distance(anchor_emb.unsqueeze(0), positive_emb.unsqueeze(0))
            neg_dist = F.pairwise_distance(anchor_emb.unsqueeze(0), negative_emb.unsqueeze(0))
            
            # 三元组损失
            triplet_loss = F.relu(pos_dist - neg_dist + self.triplet_margin)
            triplet_losses.append(triplet_loss)
        
        return torch.stack(triplet_losses).mean()
    
    def forward(self, x: torch.Tensor, ground_truth: Optional[torch.Tensor] = None, 
                return_aux_info: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 符合设计文档要求
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            ground_truth: 真实值 [batch_size, pred_len]（训练时用于三元组挖掘）
            return_aux_info: 是否返回辅助信息
            
        Returns:
            字典包含预测结果和辅助信息
        """
        batch_size = x.size(0)
        
        # === 1. Flow模型潜在表示映射 ===
        # 展平输入用于Flow模型
        x_flat = x.view(batch_size, -1)  # [batch_size, seq_len * input_dim]
        
        # 通过Flow模型映射到潜在空间
        if self.use_pretrained_flow and not self.training:
            # 推理时使用预训练的Flow模型
            with torch.no_grad():
                z_latent = self.flow_model.encode(x_flat)
        else:
            # 训练时或未使用预训练模型时
            z_latent = self.flow_model.encode(x_flat)
        
        # === 2. 门控网络计算专家权重（接受潜在表示） ===
        gating_output = self.gating(z_latent)
        expert_weights = F.softmax(gating_output / self.temperature, dim=-1)
        
        # === Top-k 稀疏激活 ===
        top_k = self.config['model'].get('top_k', None)
        if top_k is not None and top_k < self.num_experts:
            # 为每个样本保留 top_k 权重，其余置零
            topk_values, topk_indices = expert_weights.topk(top_k, dim=-1)
            mask = torch.zeros_like(expert_weights)
            mask.scatter_(1, topk_indices, 1.0)
            expert_weights = expert_weights * mask
            expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # === 3. 强制专家多样性 ===
        if self.force_diversity:
            expert_weights = self.inject_diversity_noise(expert_weights)
        
        # === 4. 专家预测 ===
        expert_predictions = []
        for i, expert in enumerate(self.experts):
            expert_pred = expert(x)  # 专家网络现在输出 [batch_size, pred_len, input_dim]
            expert_predictions.append(expert_pred)
        
        expert_predictions = torch.stack(expert_predictions, dim=1)  # [batch_size, num_experts, pred_len, input_dim]
        
        # === 5. 加权融合 ===
        # 扩展专家权重以匹配输出维度
        expert_weights_expanded = expert_weights.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_experts, 1, 1]
        weighted_predictions = torch.sum(
            expert_predictions * expert_weights_expanded, 
            dim=1
        )  # [batch_size, pred_len, input_dim]
        
        # === 6. 辅助信息和损失计算 ===
        aux_info = {}
        if return_aux_info or self.training:
            # 计算专家特征
            expert_features = self.compute_expert_features(x)
            
            # 计算原型分离损失
            prototype_loss = self.compute_prototype_separation_loss(expert_features, expert_weights)
            
            # 计算负载均衡损失
            expert_usage = expert_weights.mean(dim=0)
            load_balance_loss = torch.var(expert_usage) * self.num_experts
            
            # 计算Flow重构损失
            x_reconstructed = self.flow_model.reconstruct(x_flat)
            reconstruction_loss = F.mse_loss(x_reconstructed, x_flat)
            
            # 计算基于预测性能的三元组损失
            triplet_loss = torch.tensor(0.0, device=x.device)
            if ground_truth is not None and self.training:
                # 获取门控网络的嵌入
                gating_embeddings = self.gating.get_embeddings(z_latent)
                
                # 挖掘三元组
                triplets = self.mine_triplets_based_on_prediction_performance(
                    x, expert_weights, expert_predictions, ground_truth
                )
                
                # 计算三元组损失
                triplet_loss = self.compute_triplet_loss(gating_embeddings, triplets)
            
            # 为一致性损失提供嵌入
            aux_info['gating_embeddings'] = gating_embeddings if ground_truth is not None else self.gating.get_embeddings(z_latent)
            
            aux_info.update({
                'expert_weights': expert_weights,
                'expert_predictions': expert_predictions,
                'expert_usage': expert_usage,
                'prototype_loss': prototype_loss,
                'load_balance_loss': load_balance_loss,
                'reconstruction_loss': reconstruction_loss,
                'triplet_loss': triplet_loss,
                'temperature': self.temperature,
                'expert_features': expert_features,
                'latent_representation': z_latent
            })
        
        return {
            'predictions': weighted_predictions,
            'aux_info': aux_info
        }
    
    def get_expert_analysis(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取专家分析信息"""
        with torch.no_grad():
            output = self.forward(x, return_aux_info=True)
            aux_info = output['aux_info']
            
            # 计算专家多样性指标
            expert_weights = aux_info['expert_weights']
            expert_predictions = aux_info['expert_predictions']
            
            # 专家使用分布
            expert_usage = expert_weights.mean(dim=0)
            usage_entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
            
            # 专家预测多样性
            pred_diversity = torch.std(expert_predictions, dim=1).mean()
            
            # 专家权重分布
            weight_entropy = -torch.sum(
                expert_weights * torch.log(expert_weights + 1e-8), 
                dim=1
            ).mean()
            
            return {
                'expert_usage': expert_usage,
                'usage_entropy': usage_entropy,
                'prediction_diversity': pred_diversity,
                'weight_entropy': weight_entropy,
                'temperature': self.temperature,
                'expert_weights': expert_weights,
                'expert_predictions': expert_predictions
            }
    
    def get_config(self) -> Dict:
        """获取模型配置"""
        return {
            'model_type': 'M2_MOEP',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_experts': self.num_experts,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'temperature': self.temperature,
            'diversity_config': self.diversity_config
        }