import os
import sys
import json
import argparse
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

from configs.config_generator import ConfigGenerator
from universal_experiment import UniversalExperiment

class ExperimentRunner:
    """实验批量运行器"""
    
    def __init__(self, data_path: str = 'dataset', max_workers: int = 1):
        self.data_path = data_path
        self.max_workers = max_workers
        self.setup_logging()
        
        self.experiment = UniversalExperiment(base_data_path=data_path)

    def setup_logging(self):
        """设置日志"""
        log_file = f'experiment_runner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def run_dataset_experiment(self, dataset_name: str, **kwargs) -> Dict:
        """运行单个数据集实验（用于并行执行）"""
        try:
            result = self.experiment.run_single_experiment(dataset_name, **kwargs)
            return {
                'dataset': dataset_name,
                'status': 'success',
                'result': result
            }
        except Exception as e:
            return {
                'dataset': dataset_name,
                'status': 'failed',
                'error': str(e)
            }

    def run_experiments_sequential(self, datasets: List[str], **kwargs) -> Dict:
        """顺序运行实验"""
        self.logger.info(f"开始顺序运行 {len(datasets)} 个数据集的实验")
        
        results = {}
        for i, dataset_name in enumerate(datasets):
            self.logger.info(f"运行实验 {i+1}/{len(datasets)}: {dataset_name}")
            
            result = self.run_dataset_experiment(dataset_name, **kwargs)
            results[dataset_name] = result
            
            if result['status'] == 'success':
                self.logger.info(f"✅ {dataset_name} 实验成功完成")
            else:
                self.logger.error(f"❌ {dataset_name} 实验失败: {result['error']}")
        
        return results

    def run_experiments_parallel(self, datasets: List[str], **kwargs) -> Dict:
        """并行运行实验"""
        self.logger.info(f"开始并行运行 {len(datasets)} 个数据集的实验 (workers: {self.max_workers})")
        
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_dataset = {
                executor.submit(self.run_dataset_experiment, dataset, **kwargs): dataset
                for dataset in datasets
            }
            
            # 收集结果
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    results[dataset] = result
                    
                    if result['status'] == 'success':
                        self.logger.info(f"✅ {dataset} 实验成功完成")
                    else:
                        self.logger.error(f"❌ {dataset} 实验失败: {result['error']}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {dataset} 实验执行异常: {e}")
                    results[dataset] = {
                        'dataset': dataset,
                        'status': 'failed',
                        'error': str(e)
                    }
        
        return results

    def run_all_experiments(self, parallel: bool = False, **kwargs) -> Dict:
        """运行所有可用数据集的实验"""
        # 获取可用数据集
        available_datasets = self.experiment.list_available_datasets()
        dataset_names = [ds['name'] for ds in available_datasets]
        
        if not dataset_names:
            self.logger.warning("没有找到可用的数据集")
            return {}
        
        self.logger.info(f"找到 {len(dataset_names)} 个可用数据集: {', '.join(dataset_names)}")
        
        # 选择运行方式
        if parallel and self.max_workers > 1:
            results = self.run_experiments_parallel(dataset_names, **kwargs)
        else:
            results = self.run_experiments_sequential(dataset_names, **kwargs)
        
        # 保存汇总结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f'experiment_summary_{timestamp}.json'
        
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"实验汇总结果已保存到: {summary_file}")
        
        # 生成简要报告
        self.generate_summary_report(results, f'experiment_report_{timestamp}.txt')
        
        return results

    def run_specific_experiments(self, datasets: List[str], parallel: bool = False, **kwargs) -> Dict:
        """运行指定数据集的实验"""
        self.logger.info(f"运行指定数据集的实验: {', '.join(datasets)}")
        
        # 验证数据集是否可用
        available_datasets = self.experiment.list_available_datasets()
        available_names = [ds['name'] for ds in available_datasets]
        
        valid_datasets = []
        for dataset in datasets:
            if dataset in available_names:
                valid_datasets.append(dataset)
            else:
                self.logger.warning(f"数据集 {dataset} 不可用，跳过")
        
        if not valid_datasets:
            self.logger.error("没有有效的数据集可以运行")
            return {}
        
        # 选择运行方式
        if parallel and self.max_workers > 1:
            results = self.run_experiments_parallel(valid_datasets, **kwargs)
        else:
            results = self.run_experiments_sequential(valid_datasets, **kwargs)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f'specific_experiment_summary_{timestamp}.json'
        
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"指定实验汇总结果已保存到: {summary_file}")
        
        return results

    def generate_summary_report(self, results: Dict, save_path: str):
        """生成实验汇总报告"""
        with open(save_path, 'w') as f:
            f.write("M²-MOEP 批量实验汇总报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总实验数: {len(results)}\n\n")
            
            # 统计成功/失败
            success_count = sum(1 for r in results.values() if r['status'] == 'success')
            failed_count = len(results) - success_count
            
            f.write(f"成功实验: {success_count}\n")
            f.write(f"失败实验: {failed_count}\n\n")
            
            # 成功实验详情
            if success_count > 0:
                f.write("成功实验详情:\n")
                f.write("-" * 40 + "\n")
                for dataset, result in results.items():
                    if result['status'] == 'success':
                        metrics = result['result'].get('metrics', {})
                        f.write(f"{dataset}:\n")
                        f.write(f"  RMSE: {metrics.get('RMSE', 'N/A'):.4f}\n")
                        f.write(f"  MAE: {metrics.get('MAE', 'N/A'):.4f}\n")
                        f.write(f"  R²: {metrics.get('R2', 'N/A'):.4f}\n")
                        f.write(f"  专家熵: {metrics.get('expert_entropy', 'N/A'):.4f}\n\n")
            
            # 失败实验详情
            if failed_count > 0:
                f.write("失败实验详情:\n")
                f.write("-" * 40 + "\n")
                for dataset, result in results.items():
                    if result['status'] == 'failed':
                        f.write(f"{dataset}: {result['error']}\n")
        
        self.logger.info(f"实验汇总报告已保存到: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='批量实验运行器')
    parser.add_argument('--data-path', type=str, default='dataset', help='数据集路径')
    parser.add_argument('--datasets', nargs='+', help='指定要运行的数据集列表')
    parser.add_argument('--all', action='store_true', help='运行所有可用数据集')
    parser.add_argument('--parallel', action='store_true', help='并行运行实验')
    parser.add_argument('--workers', type=int, default=1, help='并行工作进程数')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    parser.add_argument('--num-experts', type=int, help='专家数量')
    
    args = parser.parse_args()
    
    # 构建实验参数
    experiment_kwargs = {}
    if args.epochs:
        experiment_kwargs['epochs'] = args.epochs
    if args.batch_size:
        experiment_kwargs['training'] = experiment_kwargs.get('training', {})
        experiment_kwargs['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        experiment_kwargs['training'] = experiment_kwargs.get('training', {})
        experiment_kwargs['training']['learning_rate'] = args.learning_rate
    if args.num_experts:
        experiment_kwargs['model'] = experiment_kwargs.get('model', {})
        experiment_kwargs['model']['num_experts'] = args.num_experts
    
    # 创建实验运行器
    runner = ExperimentRunner(data_path=args.data_path, max_workers=args.workers)
    
    if args.all:
        # 运行所有实验
        runner.run_all_experiments(parallel=args.parallel, **experiment_kwargs)
    elif args.datasets:
        # 运行指定实验
        runner.run_specific_experiments(args.datasets, parallel=args.parallel, **experiment_kwargs)
    else:
        # 显示帮助
        parser.print_help()
        print("\n可用数据集:")
        available = runner.experiment.list_available_datasets()
        for ds in available:
            print(f"  - {ds['name']}: {ds['description']}")

if __name__ == '__main__':
    main()
