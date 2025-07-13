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
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
import argparse
import json

from models.m2_moep import M2_MOEP
from data.universal_dataset import UniversalDataModule
from utils.metrics import calculate_metrics
from configs.config_generator import ConfigGenerator


class OutputManager:
    """
    统一的输出管理器
    负责创建和管理所有输出目录
    """
    
    def __init__(self, experiment_name: str = None, base_output_dir: str = "output"):
        """
        初始化输出管理器
        
        Args:
            experiment_name: 实验名称，如果为None则自动生成
            base_output_dir: 基础输出目录
        """
        self.base_output_dir = base_output_dir
        
        # 生成实验名称
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(base_output_dir, experiment_name)
        
        # 创建目录结构
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """创建完整的目录结构"""
        directories = [
            self.experiment_dir,
            self.get_checkpoints_dir(),
            self.get_logs_dir(),
            self.get_evaluation_dir(),
            self.get_configs_dir(),
            self.get_plots_dir()
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f" 创建实验目录: {self.experiment_dir}")
    
    def get_checkpoints_dir(self) -> str:
        """获取检查点目录"""
        return os.path.join(self.experiment_dir, "checkpoints")
    
    def get_logs_dir(self) -> str:
        """获取日志目录"""
        return os.path.join(self.experiment_dir, "logs")
    
    def get_evaluation_dir(self) -> str:
        """获取评估结果目录"""
        return os.path.join(self.experiment_dir, "evaluation")
    
    def get_configs_dir(self) -> str:
        """获取配置文件目录"""
        return os.path.join(self.experiment_dir, "configs")
    
    def get_plots_dir(self) -> str:
        """获取图表目录"""
        return os.path.join(self.experiment_dir, "plots")
    
    def get_experiment_info_path(self) -> str:
        """获取实验信息文件路径"""
        return os.path.join(self.experiment_dir, "experiment_info.json")
    
    def save_experiment_info(self, config: Dict, model_info: Dict = None):
        """保存实验信息"""
        info = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "config": config,
            "model_info": model_info,
            "directory_structure": {
                "checkpoints": self.get_checkpoints_dir(),
                "logs": self.get_logs_dir(),
                "evaluation": self.get_evaluation_dir(),
                "configs": self.get_configs_dir(),
                "plots": self.get_plots_dir()
            }
        }
        
        with open(self.get_experiment_info_path(), 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        print(f"💾 实验信息已保存到: {self.get_experiment_info_path()}")


class M2MOEPTrainer:
    """
    M²-MOEP训练器
    现在使用统一的输出管理系统
    """
    
    def __init__(self, config: Dict, output_manager: OutputManager = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化输出管理器
        if output_manager is None:
            experiment_name = config.get('experiment_name', self._generate_experiment_name())
            output_manager = OutputManager(experiment_name)
        
        self.output_manager = output_manager
        
        # 基础设置
        self.set_seed(config.get('seed', 42))
        self.setup_logging()
        
        # 保存配置文件
        self.save_config()
        
        # 初始化数据和模型
        self.data_module = UniversalDataModule(config)
        
        # 根据实际数据更新配置
        actual_features = self.data_module.get_dataset_info()['num_features']
        self.config['model']['input_dim'] = actual_features
        self.config['model']['output_dim'] = actual_features
        
        # 初始化模型
        self.model = M2_MOEP(config).to(self.device)
        
        # 保存实验信息
        model_info = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'actual_features': actual_features
        }
        self.output_manager.save_experiment_info(self.config, model_info)
        
        # 优化器和调度器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-4)
        )
        
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
        
        self.logger.info(f" 训练器初始化完成，设备: {self.device}")
        self.logger.info(f" 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f" 实验目录: {self.output_manager.experiment_dir}")
    
    def _generate_experiment_name(self) -> str:
        """生成实验名称"""
        dataset_name = self.config.get('data', {}).get('dataset_name', 'unknown')
        pred_len = self.config.get('model', {}).get('pred_len', 96)
        timestamp = datetime.now().strftime('%m%d_%H%M')
        return f"M2MOEP_{dataset_name}_pred{pred_len}_{timestamp}"
    
    def set_seed(self, seed: int):
        """设置随机种子，确保实验可重复"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.output_manager.get_logs_dir(), f'train_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f" 日志文件: {log_file}")
    
    def save_config(self):
        """保存配置文件"""
        config_path = os.path.join(self.output_manager.get_configs_dir(), 'training_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f" 配置文件已保存: {config_path}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个完整的epoch"""
        self.model.train()
        train_loader = self.data_module.get_train_loader()
        
        epoch_losses = {
            'total': 0.0,
            'prediction': 0.0,
            'reconstruction': 0.0,
            'triplet': 0.0,
            'prototype': 0.0,
            'load_balance': 0.0
        }
        
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f'训练 Epoch {self.current_epoch}') as pbar:
            for batch_idx, (batch_x, batch_y) in enumerate(pbar):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                output = self.model(batch_x, ground_truth=batch_y, return_details=True)
                
                # 计算损失
                losses = self.model.compute_loss(output, batch_y, self.current_epoch)
                total_loss = losses['total']
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training'].get('gradient_clip', 1.0)
                )
                
                self.optimizer.step()
                
                # 累积损失
                for key, loss in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key] += loss.item() if isinstance(loss, torch.Tensor) else loss
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'Pred': f"{losses.get('prediction', 0):.4f}",
                    'Triplet': f"{losses.get('triplet', 0):.4f}",
                    'Temp': f"{self.model.temperature.item():.3f}"
                })
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, current_epoch: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """验证一个epoch"""
        self.model.eval()
        val_loader = self.data_module.get_val_loader()
        
        epoch_losses = {
            'total': 0.0,
            'prediction': 0.0,
            'reconstruction': 0.0,
            'triplet': 0.0,
            'prototype': 0.0,
            'load_balance': 0.0
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                output = self.model(batch_x, return_details=True)
                losses = self.model.compute_loss(output, batch_y, current_epoch)
                
                # 累积损失
                for key, loss in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key] += loss.item() if isinstance(loss, torch.Tensor) else loss
                
                # 收集预测结果
                all_predictions.append(output['predictions'].cpu())
                all_targets.append(batch_y.cpu())
        
        # 计算平均损失
        num_batches = len(val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # 计算验证指标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        
        return epoch_losses, metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """保存训练检查点"""
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
        latest_path = os.path.join(self.output_manager.get_checkpoints_dir(), 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = os.path.join(self.output_manager.get_checkpoints_dir(), 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f" 保存最佳模型到: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载训练检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f" 检查点文件不存在: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f" 从检查点恢复训练: {checkpoint_path}")
    
    def train(self):
        """完整的训练流程"""
        self.logger.info(" 开始训练...")
        
        epochs = self.config['training']['epochs']
        patience = self.config['training'].get('patience', 20)
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练和验证
            train_losses = self.train_epoch()
            val_losses, val_metrics = self.validate_epoch(self.current_epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录训练历史
            self.training_history['train_loss'].append(train_losses['total'])
            self.training_history['val_loss'].append(val_losses['total'])
            self.training_history['metrics'].append(val_metrics)
            
            # 打印训练信息
            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
                f"RMSE: {val_metrics['RMSE']:.4f} | "
                f"MAE: {val_metrics['MAE']:.4f} | "
                f"R²: {val_metrics['R2']:.4f}"
            )
            
            # 检查是否是最佳模型
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.logger.info(f" 发现更好的模型，验证损失: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(is_best)
            
            # 早停检查
            if self.patience_counter >= patience:
                self.logger.info(f"⏹ 早停触发，在第 {epoch} 轮停止训练")
                break
        
        self.logger.info(" 训练完成！")
        self.save_training_results()
    
    def save_training_results(self):
        """保存训练结果摘要"""
        results = {
            'experiment_name': self.output_manager.experiment_name,
            'config': self.config,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'completed_at': datetime.now().isoformat()
        }
        
        # 保存到检查点目录
        results_path = os.path.join(self.output_manager.get_checkpoints_dir(), 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存最终配置
        final_config_path = os.path.join(self.output_manager.get_configs_dir(), 'final_config.yaml')
        with open(final_config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f" 训练结果已保存到: {results_path}")
        self.logger.info(f" 最终配置已保存到: {final_config_path}")
    
    def get_model(self) -> M2_MOEP:
        """获取训练好的模型"""
        return self.model
    
    def get_data_module(self) -> UniversalDataModule:
        """获取数据模块"""
        return self.data_module
    
    def get_output_manager(self) -> OutputManager:
        """获取输出管理器"""
        return self.output_manager


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='M²-MOEP训练脚本')
    
    # 实验管理
    parser.add_argument('--experiment-name', type=str, help='实验名称')
    parser.add_argument('--output-dir', type=str, default='output', help='输出根目录')
    
    # 配置相关
    parser.add_argument('--config', type=str, help='配置文件路径或数据集名称')
    parser.add_argument('--dataset', type=str, help='数据集名称')
    
    # 数据参数
    parser.add_argument('--seq-len', type=int, help='输入序列长度')
    parser.add_argument('--pred-len', type=int, help='预测长度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    
    # 模型参数
    parser.add_argument('--hidden-dim', type=int, help='隐藏层维度')
    parser.add_argument('--num-experts', type=int, help='专家数量')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 生成或加载配置
    config = None
    
    if args.config:
        if args.config.endswith('.yaml'):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = ConfigGenerator.generate_config(args.config)
    elif args.dataset:
        config = ConfigGenerator.generate_config(args.dataset)
    else:
        print(" 请指定 --config 或 --dataset 参数")
        return 1
    
    # 应用命令行参数覆盖
    if args.seq_len:
        config['model']['seq_len'] = args.seq_len
        config['data']['seq_len'] = args.seq_len
    if args.pred_len:
        config['model']['pred_len'] = args.pred_len
        config['data']['pred_len'] = args.pred_len
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.hidden_dim:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.num_experts:
        config['model']['num_experts'] = args.num_experts
    if args.seed:
        config['seed'] = args.seed
    
    # 设置实验名称
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    
    try:
        # 创建输出管理器
        output_manager = OutputManager(
            experiment_name=config.get('experiment_name'),
            base_output_dir=args.output_dir
        )
        
        # 创建训练器
        trainer = M2MOEPTrainer(config, output_manager)
        
        # 如果需要恢复训练
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        trainer.train()
        
        print(f"\n 训练完成！")
        print(f" 所有输出文件保存在: {output_manager.experiment_dir}")
        print(f" 检查点: {output_manager.get_checkpoints_dir()}")
        print(f" 日志: {output_manager.get_logs_dir()}")
        print(f" 配置: {output_manager.get_configs_dir()}")
        
    except Exception as e:
        print(f" 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    main()