#!/usr/bin/env python3
"""
M²-MOEP训练脚本
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
import argparse
import json

from models.m2_moep import M2_MOEP
from data.universal_dataset import UniversalDataModule
from utils.losses import CompositeLoss
from utils.metrics import calculate_metrics
from configs.config_generator import ConfigGenerator


class M2MOEPTrainer:
    """M²-MOEP训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 配置验证
        self._validate_config()
        
        # 设置随机种子
        self.set_seed(config.get('seed', 42))
        
        # 初始化日志
        self.setup_logging()
        
        # 确保Flow模型存在
        self._ensure_flow_model()
        
        # 初始化数据模块
        self.data_module = UniversalDataModule(config)
        
        # 更新配置中的实际特征数
        actual_features = self.data_module.get_dataset_info()['num_features']
        self.config['model']['input_dim'] = actual_features
        self.config['model']['output_dim'] = actual_features
        
        # 确保门控网络配置包含必要的维度信息
        if 'embedding_dim' not in self.config['model']:
            self.config['model']['embedding_dim'] = min(128, actual_features)
        
        # 初始化模型
        self.model = M2_MOEP(config).to(self.device)
        
        # 🔧 修复：不再使用独立的CompositeLoss，使用模型内置的损失计算
        # self.criterion = CompositeLoss(config)  # 注释掉
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-4)
        )
        
        # 初始化学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training'].get('min_lr', 1e-6)
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        # 梯度监控变量
        self.gradient_stats = {
            'grad_norms': [],
            'grad_clip_count': 0,
            'max_grad_norm': 0.0,
            'adaptive_clip_value': self.config['training'].get('gradient_clip', 1.0)
        }
        
        # 添加梯度裁剪阈值属性
        self.grad_clip_threshold = self.config['training'].get('gradient_clip', 1.0)
        
        # 创建保存目录
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.logger.info(f"训练器初始化完成，设备: {self.device}")
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 训练配置 - 添加默认值
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 20)
        self.batch_size = train_config.get('batch_size', 16)
        self.learning_rate = train_config.get('learning_rate', 0.001)
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _validate_config(self):
        """验证配置文件的有效性"""
        # 验证模型配置
        model_config = self.config.get('model', {})
        if model_config.get('input_dim', 0) <= 0:
            raise ValueError("模型输入维度必须大于0")
        
        if model_config.get('hidden_dim', 0) <= 0:
            raise ValueError("模型隐藏层维度必须大于0")
        
        if model_config.get('output_dim', 0) <= 0:
            raise ValueError("模型输出维度必须大于0")
        
        if model_config.get('num_experts', 0) <= 0:
            raise ValueError("专家数量必须大于0")
        
        if model_config.get('seq_len', 0) <= 0:
            raise ValueError("序列长度必须大于0")
        
        if model_config.get('pred_len', 0) <= 0:
            raise ValueError("预测长度必须大于0")
        
        # 验证训练配置
        train_config = self.config.get('training', {})
        if train_config.get('batch_size', 0) <= 0:
            raise ValueError("批次大小必须大于0")
        
        if train_config.get('learning_rate', 0) <= 0:
            raise ValueError("学习率必须大于0")
        
        if train_config.get('epochs', 0) <= 0:
            raise ValueError("训练轮数必须大于0")
        
        # 验证数据配置
        data_config = self.config.get('data', {})
        if not data_config.get('dataset_name'):
            raise ValueError("数据集名称不能为空")
        
        # 验证梯度裁剪配置
        gradient_clip = train_config.get('gradient_clip', 1.0)
        if gradient_clip <= 0:
            raise ValueError("梯度裁剪阈值必须大于0")
        
        # 验证Flow模型配置
        flow_config = model_config.get('flow', {})
        if flow_config.get('latent_dim', 0) <= 0:
            raise ValueError("Flow潜在维度必须大于0")
        
        # 验证温度配置
        temp_config = model_config.get('temperature', {})
        if temp_config.get('initial', 1.0) <= 0:
            raise ValueError("初始温度必须大于0")
        
        if temp_config.get('min', 0.1) <= 0:
            raise ValueError("最小温度必须大于0")
        
        if temp_config.get('max', 10.0) <= 0:
            raise ValueError("最大温度必须大于0")
        
        # 验证损失权重配置
        loss_weights = train_config.get('loss_weights', {})
        for key, value in loss_weights.items():
            if key.startswith('init_sigma_') and value <= 0:
                raise ValueError(f"损失权重初始化参数{key}必须大于0")
        
        print("配置验证通过 ✓")
    
    def setup_logging(self):
        """设置日志"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'train_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _ensure_flow_model(self):
        """确保Flow模型存在；若不存在则自动预训练"""
        flow_model_path = self.config['training'].get('flow_model_path', 'flow_model_default.pth')
        
        if os.path.exists(flow_model_path):
            print(f"Flow模型已存在: {flow_model_path}")
            return
        
        print(f"Flow模型未找到，开始自动预训练: {flow_model_path}")
        
        try:
            # 导入预训练函数
            from pretrain_flow import pretrain_flow_model
            success = pretrain_flow_model(self.config, flow_model_path)
            
            if success:
                print(f"Flow模型预训练完成: {flow_model_path}")
            else:
                print(f"Flow模型预训练失败，将继续训练但不使用Flow重构")
                
        except Exception as e:
            print(f"Flow模型预训练失败: {e}")
            print("将继续训练，但不使用Flow重构损失")
    
    def _check_layer_gradients(self):
        """检查每层的梯度范数，用于调试"""
        layer_grad_norms = {}
        problematic_layers = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                layer_grad_norms[name] = grad_norm
                
                # 检查异常梯度
                if torch.isnan(param.grad).any():
                    problematic_layers.append(f"{name}: NaN梯度")
                elif torch.isinf(param.grad).any():
                    problematic_layers.append(f"{name}: Inf梯度")
                elif grad_norm > 10.0:
                    problematic_layers.append(f"{name}: 梯度过大({grad_norm:.4f})")
        
        # 输出异常信息
        if problematic_layers:
            self.logger.warning(f"发现异常梯度: {'; '.join(problematic_layers)}")
        
        # 输出前5个最大梯度范数的层
        top_layers = sorted(layer_grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        grad_info = [f"{name}: {norm:.4f}" for name, norm in top_layers]
        self.logger.debug(f"梯度范数最大的5层: {'; '.join(grad_info)}")
    
    def _adaptive_gradient_clipping(self, losses: Dict[str, torch.Tensor]) -> float:
        """
        自适应梯度裁剪 - 修复版本
        :param losses: 损失字典
        :return: 使用的裁剪阈值
        """
        # 计算总梯度范数
        total_norm = 0.0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** 0.5
        
        # 修复梯度裁剪策略
        if total_norm > self.grad_clip_threshold:
            # 不要将阈值降到太低
            min_threshold = 0.1  # 最小阈值
            
            # 根据梯度爆炸严重程度调整
            if total_norm > 10.0:
                # 严重梯度爆炸
                self.grad_clip_threshold = max(min_threshold, self.grad_clip_threshold * 0.5)
            elif total_norm > 5.0:
                # 中等梯度爆炸
                self.grad_clip_threshold = max(min_threshold, self.grad_clip_threshold * 0.8)
            else:
                # 轻微梯度爆炸
                self.grad_clip_threshold = max(min_threshold, self.grad_clip_threshold * 0.9)
            
            self.logger.warning(f"检测到梯度爆炸: {total_norm:.4f}")
            self.logger.info(f"调整梯度裁剪阈值为: {self.grad_clip_threshold:.4f}")
            
            # 应用梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_threshold)
        
        else:
            # 梯度正常时，逐渐恢复阈值
            if self.grad_clip_threshold < 1.0:
                self.grad_clip_threshold = min(1.0, self.grad_clip_threshold * 1.01)
        
        return self.grad_clip_threshold
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        train_loader = self.data_module.get_train_loader()
        
        epoch_losses = {
            'total': 0.0,
            'prediction': 0.0,
            'reconstruction': 0.0,
            'triplet': 0.0,
            'contrastive': 0.0,
            'consistency': 0.0,
            'load_balance': 0.0,
            'prototype': 0.0
        }
        
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                if len(batch_data) == 2:
                    batch_x, batch_y = batch_data
                else:
                    print(f"配置错误: 期望2个值，但得到{len(batch_data)}个")
                    continue
                
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # 检查模型是否有部分组件在CUDA上（如Mamba专家）
                model_has_cuda_components = any(
                    hasattr(expert, 'use_mamba') and expert.use_mamba 
                    for expert in self.model.experts if hasattr(expert, 'use_mamba')
                )
                
                if model_has_cuda_components and torch.cuda.is_available():
                    # 如果模型有CUDA组件，确保数据也在CUDA上
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                
                # 前向传播
                self.optimizer.zero_grad()
                
                # 模型预测 - 传递ground_truth用于三元组挖掘
                output = self.model(batch_x, ground_truth=batch_y, return_aux_info=True)
                
                # 🔧 修复：移除Flow重构代码，模型内部处理
                # predictions = output['predictions']
                # aux_info = output['aux_info']
                
                # 获取Flow重构（如果可用）
                # reconstructed = None
                # if hasattr(self.data_module, 'get_flow_reconstruction'):
                #     reconstructed = self.data_module.get_flow_reconstruction(batch_x)
                #     if reconstructed is not None and reconstructed.numel() > 0:
                #         reconstructed = reconstructed.to(self.device)
                #         aux_info['original_input'] = batch_x
                
                # 计算损失
                losses = self.model.compute_loss(
                    outputs=output,
                    targets=batch_y,
                    epoch=self.current_epoch
                )
                total_loss = losses['total']
                
                # 反向传播
                total_loss.backward()
                
                # 增强的梯度裁剪和监控
                # 1. 先计算梯度范数（不进行裁剪）
                total_grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                # 2. 梯度统计和监控
                self.gradient_stats['grad_norms'].append(total_grad_norm)
                self.gradient_stats['max_grad_norm'] = max(
                    self.gradient_stats['max_grad_norm'], 
                    total_grad_norm
                )
                
                # 3. 检查梯度异常
                if torch.isnan(torch.tensor(total_grad_norm)) or torch.isinf(torch.tensor(total_grad_norm)):
                    self.logger.warning(f"梯度范数异常: {total_grad_norm}")
                    # 跳过这个batch的参数更新
                    self.optimizer.zero_grad()
                    continue
                
                # 4. 自适应梯度裁剪 - 简化版本
                clip_value = self.grad_clip_threshold
                if total_grad_norm > clip_value:
                    # 应用梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                    
                    # 简化阈值调整
                    if total_grad_norm > 10.0:
                        self.grad_clip_threshold = max(0.1, self.grad_clip_threshold * 0.5)
                    
                    self.logger.warning(f"梯度裁剪: {total_grad_norm:.4f} -> {clip_value:.4f}")
                    self.gradient_stats['grad_clip_count'] += 1
                else:
                    # 梯度正常时，逐渐恢复阈值
                    if self.grad_clip_threshold < 1.0:
                        self.grad_clip_threshold = min(1.0, self.grad_clip_threshold * 1.01)
                
                self.optimizer.step()
                
                # 更新温度调度
                if hasattr(self.model, 'update_temperature_schedule'):
                    if 'expert_weights' in output and output['expert_weights'] is not None:
                        expert_weights = output['expert_weights']
                        if expert_weights.dim() == 2 and expert_weights.size(0) > 0:
                            expert_usage = expert_weights.mean(dim=0)
                            expert_entropy = -torch.sum(
                                expert_usage * torch.log(expert_usage + 1e-8)
                            )
                            self.model.update_temperature_schedule(self.current_epoch, expert_entropy)
                
                # 累积损失 - 简化版本
                for key, loss in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key] += loss.item() if isinstance(loss, torch.Tensor) else loss
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'Pred': f"{losses.get('prediction', 0):.4f}",
                    'Triplet': f"{losses.get('triplet', 0):.4f}",
                    'Temp': f"{self.model.temperature.item():.3f}",
                    'Grad': f"{total_grad_norm:.3f}",
                    'Clip': f"{clip_value:.3f}"
                })
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, current_epoch: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        验证单个epoch
        
        Args:
            current_epoch (int): 当前的训练轮次
            
        Returns:
            验证损失和指标
        """
        self.model.eval()
        val_loader = self.data_module.get_val_loader()
        
        epoch_losses = {
            'total': 0.0,
            'prediction': 0.0,
            'reconstruction': 0.0,
            'triplet': 0.0,
            'contrastive': 0.0,
            'consistency': 0.0,
            'load_balance': 0.0,
            'prototype': 0.0
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # 检查模型是否有部分组件在CUDA上（如Mamba专家）
                model_has_cuda_components = any(
                    hasattr(expert, 'use_mamba') and expert.use_mamba 
                    for expert in self.model.experts if hasattr(expert, 'use_mamba')
                )
                
                if model_has_cuda_components and torch.cuda.is_available():
                    # 如果模型有CUDA组件，确保数据也在CUDA上
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                
                # 验证时不需要三元组挖掘，所以不传递ground_truth
                output = self.model(batch_x, return_aux_info=True)
                
                # 🔧 修复：移除Flow重构代码，模型内部处理
                # predictions = output['predictions']
                # aux_info = output['aux_info']
                
                # 获取Flow重构
                # reconstructed = None
                # if hasattr(self.data_module, 'get_flow_reconstruction'):
                #     reconstructed = self.data_module.get_flow_reconstruction(batch_x)
                #     if reconstructed is not None and reconstructed.numel() > 0:
                #         reconstructed = reconstructed.to(self.device)
                #         aux_info['original_input'] = batch_x
                
                # 计算损失
                losses = self.model.compute_loss(
                    outputs=output,
                    targets=batch_y,
                    epoch=current_epoch
                )
                total_loss = losses['total']
                
                # 累积损失 - 简化版本
                for key, loss in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key] += loss.item() if isinstance(loss, torch.Tensor) else loss
                
                # 收集预测和目标用于计算指标
                all_predictions.append(output['predictions'].cpu())
                all_targets.append(batch_y.cpu())
        
        # 计算平均损失
        num_batches = len(val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # 计算指标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        # 更新温度调度
        if self.config['model'].get('use_temperature_scheduler', False):
            self.model.update_temperature(total_loss, current_epoch)
        
        return epoch_losses, metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"加载检查点成功: {checkpoint_path}")
    
    def train(self):
        """完整训练流程"""
        self.logger.info("开始训练...")
        
        epochs = self.config['training']['epochs']
        patience = self.config['training'].get('patience', 20)
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_losses = self.train_epoch()
            
            # 验证一个epoch
            val_losses, val_metrics = self.validate_epoch(self.current_epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录训练历史
            self.training_history['train_loss'].append(train_losses['total'])
            self.training_history['val_loss'].append(val_losses['total'])
            self.training_history['metrics'].append(val_metrics)
            
            # 日志输出
            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
                f"RMSE: {val_metrics['RMSE']:.4f} | "
                f"MAE: {val_metrics['MAE']:.4f} | "
                f"R²: {val_metrics['R2']:.4f} | "
                f"Temp: {self.model.temperature.item():.3f}"
            )
            
            # 检查是否为最佳模型
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.logger.info(f"新的最佳验证损失: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # 保存检查点
            if epoch % self.config['training'].get('save_interval', 10) == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # 早停检查
            if self.patience_counter >= patience:
                self.logger.info(f"早停触发，在epoch {epoch}")
                break
        
        self.logger.info("训练完成！")
        
        # 保存最终结果
        self.save_training_results()
    
    def save_training_results(self):
        """保存训练结果"""
        results = {
            'config': self.config,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch,
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        # 保存为JSON
        results_path = os.path.join(self.save_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存配置
        config_path = os.path.join(self.save_dir, 'final_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"训练结果保存到: {results_path}")
    
    def get_model(self) -> M2_MOEP:
        """获取训练好的模型"""
        return self.model
    
    def get_data_module(self) -> UniversalDataModule:
        """获取数据模块"""
        return self.data_module


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='M²-MOEP训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset', type=str, help='数据集名称')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # 假设是数据集名称，生成配置
        config = ConfigGenerator.generate_config(args.config)
    
    # 应用命令行参数覆盖
    if args.dataset:
        config['data']['dataset_name'] = args.dataset
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    try:
        # 创建训练器
        trainer = M2MOEPTrainer(config)
        
        # 恢复训练（如果指定）
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        trainer.train()
        
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请检查配置文件设置")
        return 1
    except Exception as e:
        print(f"训练过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    main()