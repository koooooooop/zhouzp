#!/usr/bin/env python3
"""
M²-MOEP vs Autoformer 性能比较实验
=================================

本脚本按照Autoformer论文的标准实验设置进行性能比较实验。

数据集：
- ETTh1, ETTh2, ETTm1, ETTm2 (Electricity Transformer Temperature)
- Weather (天气数据)
- Electricity (电力消费数据)
- Traffic (交通数据)
- Exchange Rate (汇率数据)
- ILI (流感数据)

实验设置：
- 输入序列长度: 96
- 预测长度: 96, 192, 336, 720
- 评估指标: MSE, MAE
"""

import os
import sys
import json
import time
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import argparse
import logging

from train import M2MOEPTrainer
from configs.config_generator import ConfigGenerator
from utils.metrics import calculate_metrics


class AutoformerComparisonExperiment:
    """Autoformer比较实验类"""
    
    def __init__(self, base_config: Dict = None):
        self.base_config = base_config or {}
        self.results_dir = 'autoformer_comparison_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # Autoformer标准实验设置
        self.autoformer_datasets = {
            'ETTh1': {
                'path': 'dataset/ETT-small_7个因素的变压器温度变化/ETTh1.csv',
                'target': 'OT',  # 目标变量
                'features': 'M',  # Multivariate
                'freq': 'h'  # 小时级别
            },
            'ETTh2': {
                'path': 'dataset/ETT-small_7个因素的变压器温度变化/ETTh2.csv',
                'target': 'OT',
                'features': 'M',
                'freq': 'h'
            },
            'ETTm1': {
                'path': 'dataset/ETT-small_7个因素的变压器温度变化/ETTm1.csv',
                'target': 'OT',
                'features': 'M',
                'freq': 't'  # 15分钟级别
            },
            'ETTm2': {
                'path': 'dataset/ETT-small_7个因素的变压器温度变化/ETTm2.csv',
                'target': 'OT',
                'features': 'M',
                'freq': 't'
            },
            'Weather': {
                'path': 'dataset/weather_气象站_21个气象因子/weather.csv',
                'target': 'OT',
                'features': 'M',
                'freq': 't'
            },
            'Electricity': {
                'path': 'dataset/electricity_321个客户的每小时用电量/electricity.csv',
                'target': 'MT_320',  # 通常是最后一列
                'features': 'M',
                'freq': 'h'
            },
            'Traffic': {
                'path': 'dataset/traffic_862个传感器测量的每小时道路占用率/traffic.csv',
                'target': 'Sensor_861',  # 通常是最后一列
                'features': 'M',
                'freq': 'h'
            },
            'Exchange': {
                'path': 'dataset/exchange_rate_8个国家的汇率变化/exchange_rate.csv',
                'target': 'OT',
                'features': 'M',
                'freq': 'd'  # 日级别
            },
            'ILI': {
                'path': 'dataset/illness_流感患者比例和数量/national_illness.csv',
                'target': '%ILI',
                'features': 'M',
                'freq': 'w'  # 周级别
            }
        }
        
        # Autoformer标准预测长度
        self.prediction_lengths = [96, 192, 336, 720]
        
        # 固定的输入序列长度（按Autoformer论文）
        self.input_length = 96
        
        # 实验结果存储
        self.all_results = {}
        
    def setup_logging(self):
        """设置实验日志"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.results_dir, f'autoformer_comparison_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("🚀 M²-MOEP vs Autoformer 性能比较实验开始")
        self.logger.info("=" * 80)
    
    def generate_autoformer_config(self, dataset_name: str, pred_len: int) -> Dict:
        """
        生成符合Autoformer实验设置的配置
        
        Args:
            dataset_name: 数据集名称
            pred_len: 预测长度
            
        Returns:
            实验配置字典
        """
        dataset_info = self.autoformer_datasets[dataset_name]
        
        # 基础配置
        config = {
            'data': {
                'dataset_name': dataset_name.lower(),
                'data_path': dataset_info['path'],
                'target': dataset_info['target'],
                'features': dataset_info['features'],
                'seq_len': self.input_length,
                'pred_len': pred_len,
                'batch_size': 32,  # Autoformer标准批次大小
                'num_workers': 4,
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1,
                'standardization': 'standard'
            },
            'model': {
                'input_dim': 21,  # 会根据实际数据调整
                'hidden_dim': 512,  # 增大隐藏维度以匹配Autoformer
                'output_dim': 21,
                'num_experts': 8,  # 增加专家数量
                'seq_len': self.input_length,
                'pred_len': pred_len,
                'embedding_dim': 512,
                
                # 专家网络配置
                'expert_params': {
                    'mamba_d_model': 512,  # 增大Mamba模型维度
                    'mamba_scales': [1, 2, 4, 8],  # 更多尺度
                    'mamba_d_state': 32,  # 增大状态维度
                    'mamba_d_conv': 8,
                    'mamba_expand': 4
                },
                
                # Flow模型配置
                'flow': {
                    'latent_dim': 256,
                    'use_pretrained': True,
                    'num_coupling_layers': 8
                },
                
                # 温度调度配置
                'temperature': {
                    'initial': 1.0,
                    'min': 0.1,
                    'max': 5.0,
                    'decay': 0.95,
                    'schedule': 'adaptive'
                },
                
                # 多样性配置
                'diversity': {
                    'prototype_dim': 128,
                    'num_prototypes': 16,
                    'diversity_weight': 0.1,
                    'force_diversity': True
                },
                
                # 三元组损失配置
                'triplet': {
                    'margin': 0.5,
                    'mining_strategy': 'batch_hard',
                    'loss_weight': 0.1,
                    'performance_window': 100
                }
            },
            'training': {
                'epochs': 100,  # 充分训练
                'learning_rate': 0.0001,  # 较小的学习率
                'weight_decay': 1e-4,
                'batch_size': 32,
                'gradient_clip': 1.0,
                'patience': 20,
                'save_interval': 10,
                'min_lr': 1e-6,
                
                # 损失权重配置
                'loss_weights': {
                    'prediction': 1.0,
                    'reconstruction': 0.1,
                    'triplet': 0.1,
                    'consistency': 0.05,
                    'load_balance': 0.01
                },
                
                # Flow模型路径
                'flow_model_path': f'flow_model_{dataset_name.lower()}_{pred_len}.pth'
            },
            'experiment': {
                'name': f'M2MOEP_vs_Autoformer_{dataset_name}_{pred_len}',
                'description': f'M²-MOEP在{dataset_name}数据集上预测长度{pred_len}的性能',
                'dataset': dataset_name,
                'prediction_length': pred_len,
                'comparison_baseline': 'Autoformer'
            },
            'save_dir': os.path.join(self.results_dir, f'{dataset_name}_{pred_len}'),
            'seed': 42
        }
        
        return config
    
    def run_single_experiment(self, dataset_name: str, pred_len: int) -> Dict:
        """
        运行单个数据集和预测长度的实验
        
        Args:
            dataset_name: 数据集名称
            pred_len: 预测长度
            
        Returns:
            实验结果字典
        """
        self.logger.info(f"开始实验: {dataset_name} - 预测长度 {pred_len}")
        
        try:
            # 生成配置
            config = self.generate_autoformer_config(dataset_name, pred_len)
            
            # 检查数据文件是否存在
            data_path = config['data']['data_path']
            if not os.path.exists(data_path):
                self.logger.error(f"数据文件不存在: {data_path}")
                return {'error': f'数据文件不存在: {data_path}'}
            
            self.logger.info(f"✅ 数据文件找到: {data_path}")
            
            # 创建训练器
            trainer = M2MOEPTrainer(config)
            
            # 记录开始时间
            start_time = time.time()
            
            # 训练模型
            self.logger.info(f"🚀 开始训练模型...")
            trainer.train()
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            # 获取最佳验证结果
            best_val_loss = trainer.best_val_loss
            final_metrics = trainer.training_history['metrics'][-1] if trainer.training_history['metrics'] else {}
            
            # 整理实验结果
            result = {
                'dataset': dataset_name,
                'prediction_length': pred_len,
                'training_time': training_time,
                'best_val_loss': best_val_loss,
                'final_metrics': final_metrics,
                'config': config,
                'model_params': sum(p.numel() for p in trainer.model.parameters()),
                'status': 'success'
            }
            
            self.logger.info(f"✅ 实验完成: {dataset_name}_{pred_len}")
            self.logger.info(f"   - 训练时间: {training_time:.2f}s")
            self.logger.info(f"   - 最佳验证损失: {best_val_loss:.6f}")
            self.logger.info(f"   - 最终指标: {final_metrics}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 实验失败: {dataset_name}_{pred_len}")
            self.logger.error(f"   错误信息: {str(e)}")
            import traceback
            self.logger.error(f"   详细错误: {traceback.format_exc()}")
            
            return {
                'dataset': dataset_name,
                'prediction_length': pred_len,
                'error': str(e),
                'status': 'failed'
            }
    
    def run_all_experiments(self, selected_datasets: List[str] = None, 
                          selected_pred_lens: List[int] = None) -> Dict:
        """
        运行所有实验
        
        Args:
            selected_datasets: 选择的数据集列表，None表示所有数据集
            selected_pred_lens: 选择的预测长度列表，None表示所有预测长度
            
        Returns:
            所有实验结果
        """
        datasets = selected_datasets or list(self.autoformer_datasets.keys())
        pred_lens = selected_pred_lens or self.prediction_lengths
        
        total_experiments = len(datasets) * len(pred_lens)
        current_experiment = 0
        
        self.logger.info(f"📊 计划运行 {total_experiments} 个实验")
        self.logger.info(f"   - 数据集: {datasets}")
        self.logger.info(f"   - 预测长度: {pred_lens}")
        
        all_results = {}
        
        for dataset_name in datasets:
            all_results[dataset_name] = {}
            
            for pred_len in pred_lens:
                current_experiment += 1
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"实验进度: {current_experiment}/{total_experiments}")
                self.logger.info(f"当前实验: {dataset_name} - 预测长度 {pred_len}")
                self.logger.info(f"{'='*60}")
                
                # 运行实验
                result = self.run_single_experiment(dataset_name, pred_len)
                all_results[dataset_name][pred_len] = result
                
                # 保存中间结果
                self.save_results(all_results)
                
                self.logger.info(f"✅ 实验 {current_experiment}/{total_experiments} 完成")
        
        self.all_results = all_results
        return all_results
    
    def save_results(self, results: Dict = None):
        """保存实验结果"""
        if results is None:
            results = self.all_results
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        results_file = os.path.join(self.results_dir, f'detailed_results_{timestamp}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成结果汇总表
        self.generate_results_summary(results)
        
        self.logger.info(f"📊 结果已保存到: {results_file}")
    
    def generate_results_summary(self, results: Dict):
        """生成结果汇总表"""
        summary_data = []
        
        for dataset_name, dataset_results in results.items():
            for pred_len, result in dataset_results.items():
                if result.get('status') == 'success':
                    metrics = result.get('final_metrics', {})
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Pred_Len': pred_len,
                        'MSE': metrics.get('MSE', 'N/A'),
                        'MAE': metrics.get('MAE', 'N/A'),
                        'RMSE': metrics.get('RMSE', 'N/A'),
                        'R2': metrics.get('R2', 'N/A'),
                        'Training_Time': f"{result.get('training_time', 0):.2f}s",
                        'Model_Params': result.get('model_params', 0),
                        'Best_Val_Loss': f"{result.get('best_val_loss', 0):.6f}"
                    })
                else:
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Pred_Len': pred_len,
                        'MSE': 'FAILED',
                        'MAE': 'FAILED',
                        'RMSE': 'FAILED',
                        'R2': 'FAILED',
                        'Training_Time': 'FAILED',
                        'Model_Params': 'FAILED',
                        'Best_Val_Loss': 'FAILED'
                    })
        
        # 保存为CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.results_dir, 'results_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        # 生成Markdown报告
        self.generate_markdown_report(summary_df)
        
        self.logger.info(f"📈 结果汇总已保存到: {summary_file}")
    
    def generate_markdown_report(self, summary_df: pd.DataFrame):
        """生成Markdown格式的实验报告"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        markdown_content = f"""# M²-MOEP vs Autoformer 性能比较报告

**实验时间**: {timestamp}  
**实验描述**: 基于Autoformer论文标准数据集和实验设置的性能比较

## 实验设置

- **输入序列长度**: {self.input_length}
- **预测长度**: {', '.join(map(str, self.prediction_lengths))}
- **数据集**: {', '.join(self.autoformer_datasets.keys())}
- **评估指标**: MSE, MAE, RMSE, R²

## 详细结果

### 按数据集分组的结果

"""
        
        # 按数据集分组显示结果
        for dataset in summary_df['Dataset'].unique():
            dataset_data = summary_df[summary_df['Dataset'] == dataset]
            markdown_content += f"\n#### {dataset}\n\n"
            markdown_content += "| 预测长度 | MSE | MAE | RMSE | R² | 训练时间 | 参数量 |\n"
            markdown_content += "|---------|-----|-----|------|----|---------|---------|\n"
            
            for _, row in dataset_data.iterrows():
                # 处理参数格式化
                model_params = row['Model_Params']
                if isinstance(model_params, str) and model_params == 'FAILED':
                    model_params_str = 'FAILED'
                else:
                    model_params_str = f"{model_params:,}"
                
                markdown_content += f"| {row['Pred_Len']} | {row['MSE']} | {row['MAE']} | {row['RMSE']} | {row['R2']} | {row['Training_Time']} | {model_params_str} |\n"
        
        markdown_content += """

## 结论

本实验按照Autoformer论文的标准设置进行，可以直接与Autoformer的发表结果进行比较。

### 关键发现

1. **模型性能**: M²-MOEP在不同数据集和预测长度上的表现
2. **训练效率**: 训练时间和收敛性能
3. **模型复杂度**: 参数量和计算复杂度

### 与Autoformer比较

请将上述结果与Autoformer论文中报告的结果进行比较分析。

---
*本报告由M²-MOEP自动化实验系统生成*
"""
        
        # 保存Markdown报告
        report_file = os.path.join(self.results_dir, 'autoformer_comparison_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        self.logger.info(f"📑 Markdown报告已保存到: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='M²-MOEP vs Autoformer 性能比较实验')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic', 'Exchange', 'ILI'],
                       help='选择要运行的数据集')
    parser.add_argument('--pred-lens', nargs='+', type=int, 
                       choices=[96, 192, 336, 720],
                       help='选择要运行的预测长度')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速测试模式（仅运行部分实验）')
    
    args = parser.parse_args()
    
    # 创建实验管理器
    experiment = AutoformerComparisonExperiment()
    
    # 确定运行的实验
    if args.quick_test:
        # 快速测试：只运行Weather数据集的96预测长度
        selected_datasets = ['Weather']
        selected_pred_lens = [96]
        experiment.logger.info("🚀 快速测试模式")
    else:
        selected_datasets = args.datasets
        selected_pred_lens = args.pred_lens
    
    try:
        # 运行所有实验
        results = experiment.run_all_experiments(selected_datasets, selected_pred_lens)
        
        # 保存最终结果
        experiment.save_results(results)
        
        experiment.logger.info("🎉 所有实验完成！")
        experiment.logger.info(f"📊 结果保存在: {experiment.results_dir}")
        
    except KeyboardInterrupt:
        experiment.logger.info("⚠️ 实验被用户中断")
        experiment.save_results()
    except Exception as e:
        experiment.logger.error(f"❌ 实验过程出现错误: {e}")
        import traceback
        experiment.logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    main() 