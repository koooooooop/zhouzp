#!/usr/bin/env python3
"""
M²-MOEP 模型评估脚本
现在使用统一的输出管理系统
"""

import os
import sys
import json
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入必要的模块
from models.m2_moep import M2_MOEP
from data.universal_dataset import UniversalDataModule
from utils.metrics import calculate_metrics, compute_expert_metrics
from configs.config_generator import ConfigGenerator


class EvaluationOutputManager:
    """
    评估输出管理器
    专门处理评估结果的输出
    """
    
    def __init__(self, experiment_dir: str = None, base_output_dir: str = "output"):
        """
        初始化评估输出管理器
        
        Args:
            experiment_dir: 现有实验目录，如果为None则创建新的
            base_output_dir: 基础输出目录
        """
        if experiment_dir and os.path.exists(experiment_dir):
            # 使用现有实验目录
            self.experiment_dir = experiment_dir
            self.experiment_name = os.path.basename(experiment_dir)
        else:
            # 创建新的评估目录
            if experiment_dir:
                self.experiment_name = os.path.basename(experiment_dir)
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.experiment_name = f"evaluation_{timestamp}"
            
            self.experiment_dir = os.path.join(base_output_dir, self.experiment_name)
        
        self.base_output_dir = base_output_dir
        self._create_evaluation_directories()
    
    def _create_evaluation_directories(self):
        """创建评估相关目录"""
        directories = [
            self.experiment_dir,
            self.get_evaluation_dir(),
            self.get_plots_dir(),
            self.get_tables_dir(),
            self.get_predictions_dir()
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f" 评估目录: {self.experiment_dir}")
    
    def get_evaluation_dir(self) -> str:
        """获取评估结果目录"""
        return os.path.join(self.experiment_dir, "evaluation")
    
    def get_plots_dir(self) -> str:
        """获取图表目录"""
        return os.path.join(self.experiment_dir, "plots")
    
    def get_tables_dir(self) -> str:
        """获取表格目录"""
        return os.path.join(self.experiment_dir, "tables")
    
    def get_predictions_dir(self) -> str:
        """获取预测结果目录"""
        return os.path.join(self.experiment_dir, "predictions")
    
    def get_summary_path(self) -> str:
        """获取评估摘要文件路径"""
        return os.path.join(self.get_evaluation_dir(), "evaluation_summary.json")


class M2MOEPEvaluator:
    """
    M²-MOEP模型评估器
    现在使用统一的输出管理系统
    """
    
    def __init__(self, checkpoint_path: str, config_path: str = None, 
                 output_manager: EvaluationOutputManager = None):
        """
        初始化评估器
        
        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径（可选）
            output_manager: 输出管理器
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化输出管理器
        if output_manager is None:
            # 尝试从检查点路径推断实验目录
            experiment_dir = self._infer_experiment_dir_from_checkpoint(checkpoint_path)
            output_manager = EvaluationOutputManager(experiment_dir)
        
        self.output_manager = output_manager
        
        # 加载配置和模型
        self.config = self._load_config()
        self.model = self._load_model()
        self.data_module = self._load_data_module()
        
        # 评估结果存储
        self.results = {}
        self.detailed_results = {}
        
        print(f" 评估器初始化完成")
        print(f"   - 设备: {self.device}")
        print(f"   - 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   - 输出目录: {self.output_manager.experiment_dir}")
    
    def _infer_experiment_dir_from_checkpoint(self, checkpoint_path: str) -> str:
        """从检查点路径推断实验目录"""
        # 假设检查点路径格式为: output/experiment_name/checkpoints/xxx.pth
        parts = checkpoint_path.split(os.sep)
        
        # 查找 checkpoints 目录的位置
        if 'checkpoints' in parts:
            checkpoints_index = parts.index('checkpoints')
            if checkpoints_index > 0:
                # 返回实验目录路径
                return os.sep.join(parts[:checkpoints_index])
        
        # 如果无法推断，返回None
        return None
    
    def _load_config(self) -> Dict:
        """加载配置"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # 从检查点加载配置
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        return checkpoint.get('config', {})
    
    def _load_model(self) -> M2_MOEP:
        """加载模型"""
        model = M2_MOEP(self.config).to(self.device)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def _load_data_module(self) -> UniversalDataModule:
        """加载数据模块"""
        return UniversalDataModule(self.config)
    
    def evaluate_on_test_set(self) -> Dict:
        """在测试集上评估模型"""
        print(" 开始在测试集上评估...")
        
        self.model.eval()
        test_loader = self.data_module.get_test_loader()
        
        # 存储所有结果
        all_predictions = []
        all_targets = []
        all_expert_weights = []
        
        # 损失累积
        total_loss = 0.0
        loss_components = {
            'prediction': 0.0,
            'reconstruction': 0.0,
            'triplet': 0.0,
            'prototype': 0.0,
            'load_balance': 0.0
        }
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(tqdm(test_loader, desc="评估测试集")):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_x, return_details=True)
                
                # 计算损失
                losses = self.model.compute_loss(outputs, batch_y)
                total_loss += losses['total'].item()
                
                for key in loss_components:
                    if key in losses:
                        loss_components[key] += losses[key].item()
                
                # 收集结果
                predictions = outputs['predictions'].cpu()
                targets = batch_y.cpu()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                # 收集专家信息
                if 'expert_weights' in outputs:
                    all_expert_weights.append(outputs['expert_weights'].cpu())
        
        # 合并所有结果
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # 计算指标
        metrics = calculate_metrics(predictions, targets)
        
        # 计算平均损失
        num_batches = len(test_loader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        # 计算专家指标
        expert_metrics = {}
        if all_expert_weights:
            expert_weights = torch.cat(all_expert_weights, dim=0)
            expert_metrics = compute_expert_metrics(expert_weights)
        
        # 构建结果
        results = {
            'test_metrics': metrics,
            'test_loss': avg_loss,
            'loss_components': loss_components,
            'expert_metrics': expert_metrics,
            'predictions': predictions.numpy(),
            'targets': targets.numpy(),
            'expert_weights': expert_weights.numpy() if all_expert_weights else None
        }
        
        self.results = results
        
        # 保存结果
        self._save_evaluation_results()
        
        print(f" 测试集评估完成")
        print(f"   - 测试样本数: {len(predictions)}")
        print(f"   - MSE: {metrics['MSE']:.6f}")
        print(f"   - MAE: {metrics['MAE']:.6f}")
        print(f"   - RMSE: {metrics['RMSE']:.6f}")
        print(f"   - MAPE: {metrics['MAPE']:.4f}%")
        print(f"   - R²: {metrics['R2']:.6f}")
        
        return results
    
    def _save_evaluation_results(self):
        """保存评估结果"""
        if not self.results:
            return
        
        # 保存详细结果
        detailed_results = {
            'checkpoint_path': self.checkpoint_path,
            'evaluation_time': datetime.now().isoformat(),
            'test_metrics': self.results['test_metrics'],
            'test_loss': self.results['test_loss'],
            'loss_components': self.results['loss_components'],
            'expert_metrics': self.results['expert_metrics'],
            'model_config': self.config.get('model', {}),
            'data_config': self.config.get('data', {})
        }
        
        # 保存到评估目录
        results_path = os.path.join(self.output_manager.get_evaluation_dir(), 'detailed_results.json')
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # 保存预测结果
        predictions_path = os.path.join(self.output_manager.get_predictions_dir(), 'predictions.npy')
        np.save(predictions_path, self.results['predictions'])
        
        targets_path = os.path.join(self.output_manager.get_predictions_dir(), 'targets.npy')
        np.save(targets_path, self.results['targets'])
        
        if self.results['expert_weights'] is not None:
            expert_weights_path = os.path.join(self.output_manager.get_predictions_dir(), 'expert_weights.npy')
            np.save(expert_weights_path, self.results['expert_weights'])
        
        print(f" 评估结果已保存到: {results_path}")
        print(f" 预测结果已保存到: {predictions_path}")
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """生成论文对比表格"""
        print(" 生成论文对比表格...")
        
        if not self.results:
            raise ValueError("请先运行evaluate_on_test_set()方法")
        
        metrics = self.results['test_metrics']
        
        # 创建对比表格
        comparison_data = {
            'Method': ['M²-MOEP (Ours)'],
            'MSE': [f"{metrics['MSE']:.6f}"],
            'MAE': [f"{metrics['MAE']:.6f}"],
            'RMSE': [f"{metrics['RMSE']:.6f}"],
            'MAPE': [f"{metrics['MAPE']:.4f}%"],
            'R²': [f"{metrics['R2']:.6f}"]
        }
        
        # 添加基线方法占位符
        baseline_methods = [
            'Autoformer', 'FEDformer', 'iTransformer', 'PatchTST', 
            'DLinear', 'TimesNet', 'ETSformer'
        ]
        
        for method in baseline_methods:
            comparison_data['Method'].append(method)
            comparison_data['MSE'].append('TBD')
            comparison_data['MAE'].append('TBD')
            comparison_data['RMSE'].append('TBD')
            comparison_data['MAPE'].append('TBD')
            comparison_data['R²'].append('TBD')
        
        df = pd.DataFrame(comparison_data)
        
        # 保存表格
        table_path = os.path.join(self.output_manager.get_tables_dir(), 'comparison_table.csv')
        df.to_csv(table_path, index=False)
        
        # 生成LaTeX表格
        latex_table = self._generate_latex_table(df)
        latex_path = os.path.join(self.output_manager.get_tables_dir(), 'comparison_table.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        print(f" 对比表格已保存:")
        print(f"   - CSV: {table_path}")
        print(f"   - LaTeX: {latex_path}")
        
        return df
    
    def _generate_latex_table(self, df: pd.DataFrame) -> str:
        """生成LaTeX表格"""
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Performance Comparison on Dataset}\n"
        latex += "\\label{tab:results}\n"
        latex += "\\begin{tabular}{lccccc}\n"
        latex += "\\toprule\n"
        latex += "Method & MSE & MAE & RMSE & MAPE & R² \\\\\n"
        latex += "\\midrule\n"
        
        for idx, row in df.iterrows():
            if idx == 0:  # 我们的方法
                latex += f"\\textbf{{{row['Method']}}} & "
                latex += f"\\textbf{{{row['MSE']}}} & "
                latex += f"\\textbf{{{row['MAE']}}} & "
                latex += f"\\textbf{{{row['RMSE']}}} & "
                latex += f"\\textbf{{{row['MAPE']}}} & "
                latex += f"\\textbf{{{row['R²']}}} \\\\\n"
            else:
                latex += f"{row['Method']} & {row['MSE']} & {row['MAE']} & {row['RMSE']} & {row['MAPE']} & {row['R²']} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def generate_plots(self):
        """生成可视化图表"""
        if not self.results:
            print(" 请先运行evaluate_on_test_set()方法")
            return
        
        print(" 生成可视化图表...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        # 1. 预测vs真实值散点图
        self._plot_predictions_vs_targets()
        
        # 2. 误差分布图
        self._plot_error_distribution()
        
        # 3. 专家权重分布图
        if self.results['expert_weights'] is not None:
            self._plot_expert_weights_distribution()
        
        # 4. 时间序列预测示例
        self._plot_time_series_examples()
        
        print(f" 图表已保存到: {self.output_manager.get_plots_dir()}")
    
    def _plot_predictions_vs_targets(self):
        """绘制预测vs真实值散点图"""
        predictions = self.results['predictions'].flatten()
        targets = self.results['targets'].flatten()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.5, s=1)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs True Values')
        
        # 添加R²信息
        r2 = self.results['test_metrics']['R2']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_manager.get_plots_dir(), 'predictions_vs_targets.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self):
        """绘制误差分布图"""
        predictions = self.results['predictions'].flatten()
        targets = self.results['targets'].flatten()
        errors = predictions - targets
        
        plt.figure(figsize=(12, 4))
        
        # 误差直方图
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        
        # 误差的正态Q-Q图
        plt.subplot(1, 2, 2)
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Errors')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_manager.get_plots_dir(), 'error_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_expert_weights_distribution(self):
        """绘制专家权重分布图"""
        expert_weights = self.results['expert_weights']
        
        plt.figure(figsize=(12, 8))
        
        # 专家使用率
        plt.subplot(2, 2, 1)
        expert_usage = expert_weights.mean(axis=0)
        plt.bar(range(len(expert_usage)), expert_usage)
        plt.xlabel('Expert Index')
        plt.ylabel('Average Usage')
        plt.title('Expert Usage Distribution')
        
        # 专家权重热力图
        plt.subplot(2, 2, 2)
        sample_weights = expert_weights[:100]  # 显示前100个样本
        plt.imshow(sample_weights.T, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Sample Index')
        plt.ylabel('Expert Index')
        plt.title('Expert Weights Heatmap')
        
        # 专家权重分布箱线图
        plt.subplot(2, 2, 3)
        plt.boxplot([expert_weights[:, i] for i in range(expert_weights.shape[1])], 
                   labels=range(expert_weights.shape[1]))
        plt.xlabel('Expert Index')
        plt.ylabel('Weight')
        plt.title('Expert Weight Distribution')
        
        # 专家使用熵
        plt.subplot(2, 2, 4)
        entropies = [-np.sum(w * np.log(w + 1e-8)) for w in expert_weights]
        plt.hist(entropies, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Entropy')
        plt.ylabel('Frequency')
        plt.title('Expert Weight Entropy Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_manager.get_plots_dir(), 'expert_weights_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series_examples(self):
        """绘制时间序列预测示例"""
        predictions = self.results['predictions']
        targets = self.results['targets']
        
        # 选择几个示例进行可视化
        num_examples = min(4, len(predictions))
        indices = np.random.choice(len(predictions), num_examples, replace=False)
        
        plt.figure(figsize=(15, 10))
        
        for i, idx in enumerate(indices):
            plt.subplot(2, 2, i+1)
            
            pred = predictions[idx]
            target = targets[idx]
            
            # 如果是多元时间序列，只显示第一个维度
            if pred.ndim > 1:
                pred = pred[:, 0]
                target = target[:, 0]
            
            time_steps = range(len(pred))
            plt.plot(time_steps, target, label='True', linewidth=2)
            plt.plot(time_steps, pred, label='Predicted', linewidth=2, linestyle='--')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.title(f'Example {i+1}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_manager.get_plots_dir(), 'time_series_examples.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_evaluation(self, generate_plots: bool = True) -> Dict:
        """运行完整的评估流程"""
        print(" 开始完整评估流程...")
        
        # 1. 测试集评估
        self.evaluate_on_test_set()
        
        # 2. 生成对比表格
        comparison_table = self.generate_comparison_table()
        
        # 3. 生成可视化图表
        if generate_plots:
            self.generate_plots()
        
        # 4. 生成评估摘要
        summary = self._generate_evaluation_summary()
        
        print("\n 完整评估流程已完成！")
        print(f" 所有结果保存在: {self.output_manager.experiment_dir}")
        print(f" 评估结果: {self.output_manager.get_evaluation_dir()}")
        print(f" 图表: {self.output_manager.get_plots_dir()}")
        print(f" 表格: {self.output_manager.get_tables_dir()}")
        
        return summary
    
    def _generate_evaluation_summary(self) -> Dict:
        """生成评估摘要"""
        summary = {
            'experiment_name': self.output_manager.experiment_name,
            'checkpoint_path': self.checkpoint_path,
            'evaluation_time': datetime.now().isoformat(),
            'test_metrics': self.results['test_metrics'],
            'test_loss': self.results['test_loss'],
            'loss_components': self.results['loss_components'],
            'expert_metrics': self.results['expert_metrics'],
            'dataset_info': {
                'dataset_name': self.config.get('data', {}).get('dataset_name', 'unknown'),
                'seq_len': self.config.get('model', {}).get('seq_len', 96),
                'pred_len': self.config.get('model', {}).get('pred_len', 96)
            },
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'num_experts': self.config.get('model', {}).get('num_experts', 4),
                'hidden_dim': self.config.get('model', {}).get('hidden_dim', 256)
            }
        }
        
        # 保存摘要
        summary_path = self.output_manager.get_summary_path()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f" 评估摘要已保存到: {summary_path}")
        
        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='M²-MOEP模型评估脚本')
    
    # 基础参数
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（可选）')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='output', help='输出根目录')
    parser.add_argument('--experiment-name', type=str, help='实验名称')
    
    # 评估选项
    parser.add_argument('--no-plots', action='store_true', help='不生成图表')
    parser.add_argument('--save-predictions', action='store_true', help='保存预测结果')
    
    args = parser.parse_args()
    
    # 检查检查点文件
    if not os.path.exists(args.checkpoint):
        print(f" 检查点文件不存在: {args.checkpoint}")
        return 1
    
    try:
        # 创建评估输出管理器
        if args.experiment_name:
            experiment_dir = os.path.join(args.output_dir, args.experiment_name)
        else:
            experiment_dir = None
        
        output_manager = EvaluationOutputManager(
            experiment_dir=experiment_dir,
            base_output_dir=args.output_dir
        )
        
        # 创建评估器
        evaluator = M2MOEPEvaluator(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            output_manager=output_manager
        )
        
        # 运行完整评估
        summary = evaluator.run_full_evaluation(generate_plots=not args.no_plots)
        
        print(f"\n 评估完成！主要结果:")
        metrics = summary['test_metrics']
        print(f"   - MSE: {metrics['MSE']:.6f}")
        print(f"   - MAE: {metrics['MAE']:.6f}")
        print(f"   - RMSE: {metrics['RMSE']:.6f}")
        print(f"   - MAPE: {metrics['MAPE']:.4f}%")
        print(f"   - R²: {metrics['R2']:.6f}")
        
        return 0
        
    except Exception as e:
        print(f" 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    main()