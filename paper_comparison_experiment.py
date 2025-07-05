#!/usr/bin/env python3
"""
论文对比实验脚本
与 "Non-autoregressive Conditional Diffusion Models for Time Series Prediction" 论文结果对比
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import argparse
import logging

from universal_experiment import UniversalExperiment
from configs.config_generator import ConfigGenerator

class PaperComparisonExperiment:
    """论文对比实验类"""
    
    def __init__(self, data_path: str = 'dataset'):
        self.data_path = data_path
        self.setup_logging()
        self.experiment = UniversalExperiment(base_data_path=data_path)
        
        # 论文中的重合数据集
        self.paper_datasets = [
            'weather',      # Weather
            'ETTm1',        # ETTm1
            'traffic',      # Traffic  
            'electricity',  # Electricity
            'ETTh1',        # ETTh1
            'exchange_rate' # Exchange
        ]
        
        # 论文基准结果 (MSE) - 从表格中提取
        self.paper_results = {
            'univariate': {
                'weather': {'TimeDiff': 0.002, 'DLinear': 0.168, 'FiLM': 0.007},
                'ETTm1': {'TimeDiff': 0.040, 'DLinear': 0.041, 'FiLM': 0.038},
                'traffic': {'TimeDiff': 0.121, 'DLinear': 0.139, 'FiLM': 0.198},
                'electricity': {'TimeDiff': 0.232, 'DLinear': 0.244, 'FiLM': 0.260},
                'ETTh1': {'TimeDiff': 0.066, 'DLinear': 0.078, 'FiLM': 0.070},
                'exchange_rate': {'TimeDiff': 0.017, 'DLinear': 0.017, 'FiLM': 0.018}
            },
            'multivariate': {
                'weather': {'TimeDiff': 0.311, 'DLinear': 0.488, 'FiLM': 0.327},
                'ETTm1': {'TimeDiff': 0.336, 'DLinear': 0.345, 'FiLM': 0.347},
                'traffic': {'TimeDiff': 0.564, 'DLinear': 0.389, 'FiLM': 0.628},
                'electricity': {'TimeDiff': 0.193, 'DLinear': 0.215, 'FiLM': 0.210},
                'ETTh1': {'TimeDiff': 0.407, 'DLinear': 0.445, 'FiLM': 0.426},
                'exchange_rate': {'TimeDiff': 0.018, 'DLinear': 0.022, 'FiLM': 0.016}
            }
        }

    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'paper_comparison_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def check_available_datasets(self) -> List[str]:
        """检查可用的对比数据集"""
        self.logger.info("正在检查可用的对比数据集...")
        
        available_datasets = self.experiment.list_available_datasets()
        available_names = [ds['name'] for ds in available_datasets]
        
        valid_datasets = []
        for dataset in self.paper_datasets:
            if dataset in available_names:
                valid_datasets.append(dataset)
                self.logger.info(f"✅ {dataset} - 数据集可用")
            else:
                self.logger.warning(f"❌ {dataset} - 数据集不可用")
        
        self.logger.info(f"共找到 {len(valid_datasets)} 个可用的对比数据集")
        return valid_datasets

    def run_single_comparison(self, dataset_name: str, epochs: int = 50, 
                             mode: str = 'multivariate') -> Dict:
        """运行单个数据集的对比实验"""
        self.logger.info(f"开始运行 {dataset_name} 的对比实验 (模式: {mode})")
        
        try:
            # 生成适合对比的配置
            config_kwargs = {
                'epochs': epochs,
                'seq_len': 96,
                'pred_len': 96,
                'batch_size': 32 if mode == 'multivariate' else 16
            }
            
            # 如果是单变量模式，需要特殊处理
            if mode == 'univariate':
                config_kwargs['univariate_mode'] = True
            
            result = self.experiment.run_single_experiment(dataset_name, **config_kwargs)
            
            if 'error' in result:
                return {
                    'dataset': dataset_name,
                    'mode': mode,
                    'status': 'failed',
                    'error': result['error']
                }
            
            # 提取关键指标
            metrics = result.get('metrics', {})
            
            return {
                'dataset': dataset_name,
                'mode': mode,
                'status': 'success',
                'metrics': {
                    'MSE': metrics.get('MSE', None),
                    'RMSE': metrics.get('RMSE', None),
                    'MAE': metrics.get('MAE', None),
                    'MAPE': metrics.get('MAPE', None),
                    'R2': metrics.get('R2', None)
                },
                'config': result.get('config', {}),
                'experiment_dir': result.get('experiment_dir', '')
            }
            
        except Exception as e:
            self.logger.error(f"实验失败 {dataset_name}: {str(e)}")
            return {
                'dataset': dataset_name,
                'mode': mode,
                'status': 'failed',
                'error': str(e)
            }

    def run_all_comparisons(self, epochs: int = 50, 
                           modes: List[str] = ['multivariate', 'univariate']) -> Dict:
        """运行所有对比实验"""
        self.logger.info("开始运行所有对比实验...")
        
        # 检查可用数据集
        valid_datasets = self.check_available_datasets()
        
        if not valid_datasets:
            self.logger.error("没有可用的对比数据集")
            return {}
        
        all_results = {}
        
        # 为每个模式和数据集运行实验
        for mode in modes:
            self.logger.info(f"运行 {mode} 模式的实验...")
            mode_results = {}
            
            for dataset in valid_datasets:
                self.logger.info(f"正在运行 {dataset} ({mode})")
                result = self.run_single_comparison(dataset, epochs, mode)
                mode_results[dataset] = result
                
                if result['status'] == 'success':
                    mse = result['metrics']['MSE']
                    self.logger.info(f"✅ {dataset} ({mode}) - MSE: {mse:.6f}")
                else:
                    self.logger.error(f"❌ {dataset} ({mode}) - {result['error']}")
            
            all_results[mode] = mode_results
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'paper_comparison_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self.logger.info(f"对比实验结果已保存到: {results_file}")
        
        # 生成对比报告
        report_file = f'paper_comparison_report_{timestamp}.txt'
        self.generate_comparison_report(all_results, report_file)
        
        return all_results

    def generate_comparison_report(self, results: Dict, save_path: str):
        """生成对比报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("M²-MOEP 与论文基准结果对比报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"对比论文: Non-autoregressive Conditional Diffusion Models for Time Series Prediction\n\n")
            
            # 为每个模式生成报告
            for mode in results:
                f.write(f"\n{mode.upper()} 模式结果对比\n")
                f.write("-" * 60 + "\n")
                
                # 创建对比表格
                comparison_data = []
                
                for dataset, result in results[mode].items():
                    if result['status'] == 'success':
                        our_mse = result['metrics']['MSE']
                        
                        # 获取论文基准结果
                        paper_baselines = self.paper_results.get(mode, {}).get(dataset, {})
                        
                        row = {
                            'Dataset': dataset,
                            'M²-MOEP (Ours)': f"{our_mse:.6f}" if our_mse else "N/A"
                        }
                        
                        # 添加论文基准结果
                        for method, mse in paper_baselines.items():
                            row[method] = f"{mse:.6f}"
                        
                        # 计算改进情况
                        improvements = {}
                        if our_mse:
                            for method, mse in paper_baselines.items():
                                if mse > 0:
                                    improvement = ((mse - our_mse) / mse) * 100
                                    improvements[method] = improvement
                        
                        row['improvements'] = improvements
                        comparison_data.append(row)
                
                # 写入表格
                if comparison_data:
                    # 表头
                    headers = ['Dataset', 'M²-MOEP (Ours)']
                    if comparison_data[0]:
                        paper_methods = [k for k in comparison_data[0].keys() 
                                       if k not in ['Dataset', 'M²-MOEP (Ours)', 'improvements']]
                        headers.extend(paper_methods)
                    
                    f.write(f"{'Dataset':<15}")
                    f.write(f"{'M²-MOEP':<12}")
                    for method in paper_methods:
                        f.write(f"{method:<12}")
                    f.write("改进情况\n")
                    
                    f.write("-" * 80 + "\n")
                    
                    # 数据行
                    for row in comparison_data:
                        f.write(f"{row['Dataset']:<15}")
                        f.write(f"{row['M²-MOEP (Ours)']:<12}")
                        
                        for method in paper_methods:
                            f.write(f"{row.get(method, 'N/A'):<12}")
                        
                        # 改进情况
                        improvements = row.get('improvements', {})
                        if improvements:
                            best_improvement = max(improvements.values())
                            f.write(f"最佳改进: {best_improvement:+.1f}%")
                        
                        f.write("\n")
                
                # 统计信息
                f.write(f"\n{mode.upper()} 模式统计:\n")
                successful_experiments = [r for r in results[mode].values() if r['status'] == 'success']
                f.write(f"成功实验数: {len(successful_experiments)}\n")
                f.write(f"失败实验数: {len(results[mode]) - len(successful_experiments)}\n")
                
                if successful_experiments:
                    avg_mse = np.mean([r['metrics']['MSE'] for r in successful_experiments if r['metrics']['MSE']])
                    f.write(f"平均MSE: {avg_mse:.6f}\n")
        
        self.logger.info(f"对比报告已保存到: {save_path}")

    def generate_latex_table(self, results: Dict, mode: str = 'multivariate') -> str:
        """生成LaTeX表格"""
        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{与论文基准方法的MSE对比结果}")
        latex_lines.append("\\begin{tabular}{|c|c|c|c|c|}")
        latex_lines.append("\\hline")
        latex_lines.append("Dataset & M²-MOEP (Ours) & TimeDiff & DLinear & FiLM \\\\")
        latex_lines.append("\\hline")
        
        mode_results = results.get(mode, {})
        
        for dataset, result in mode_results.items():
            if result['status'] == 'success':
                our_mse = result['metrics']['MSE']
                paper_baselines = self.paper_results.get(mode, {}).get(dataset, {})
                
                line = f"{dataset} & "
                line += f"{our_mse:.6f} & " if our_mse else "N/A & "
                line += f"{paper_baselines.get('TimeDiff', 'N/A')} & "
                line += f"{paper_baselines.get('DLinear', 'N/A')} & "
                line += f"{paper_baselines.get('FiLM', 'N/A')} \\\\"
                
                latex_lines.append(line)
        
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        return "\n".join(latex_lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行论文对比实验')
    parser.add_argument('--data-path', default='dataset', help='数据集路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--modes', nargs='+', default=['multivariate', 'univariate'], 
                       help='实验模式')
    parser.add_argument('--datasets', nargs='+', default=None, 
                       help='指定运行的数据集')
    
    args = parser.parse_args()
    
    # 创建实验对象
    comparison_exp = PaperComparisonExperiment(data_path=args.data_path)
    
    if args.datasets:
        # 运行指定数据集
        comparison_exp.logger.info(f"运行指定数据集: {args.datasets}")
        # 这里可以添加针对特定数据集的逻辑
    else:
        # 运行所有对比实验
        results = comparison_exp.run_all_comparisons(
            epochs=args.epochs,
            modes=args.modes
        )
        
        # 生成LaTeX表格
        if 'multivariate' in results:
            latex_table = comparison_exp.generate_latex_table(results, 'multivariate')
            with open('comparison_table_multivariate.tex', 'w') as f:
                f.write(latex_table)
            print("LaTeX表格已生成: comparison_table_multivariate.tex")


if __name__ == "__main__":
    main() 