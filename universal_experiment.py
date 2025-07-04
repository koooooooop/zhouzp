import os
import sys
import yaml
import json
import torch
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from configs.config_generator import ConfigGenerator
from data.universal_dataset import UniversalDataModule

class UniversalExperiment:
    """通用实验运行器"""
    
    def __init__(self, base_data_path: str = 'dataset'):
        self.base_data_path = base_data_path
        self.results_dir = 'results'
        self.setup_logging()
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)

    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.results_dir, 'experiment.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def list_available_datasets(self):
        """列出所有可用的数据集"""
        self.logger.info("可用的数据集:")
        
        available = []
        for dataset_name, config in ConfigGenerator.DATASET_CONFIGS.items():
            # 检查数据文件是否存在
            data_pattern = os.path.join(self.base_data_path, f"{dataset_name}_*")
            
            # 简单检查：查看是否有匹配的目录
            exists = False
            if os.path.exists(self.base_data_path):
                exists = len([d for d in os.listdir(self.base_data_path) if d.startswith(dataset_name)]) > 0
            
            status = "✅" if exists else "❌"
            self.logger.info(f"  {status} {dataset_name}: {config['description']}")
            
            if exists:
                available.append({
                    'name': dataset_name,
                    'description': config['description'],
                    'features': config['expected_features'],
                    'seq_len': config['seq_len'],
                    'pred_len': config['pred_len']
                })
        
        return available

    def get_dataset_summary(self, dataset_name: str) -> Dict:
        """获取数据集摘要信息"""
        if dataset_name not in ConfigGenerator.DATASET_CONFIGS:
            raise ValueError(f"未支持的数据集: {dataset_name}")
        
        config = ConfigGenerator.generate_config(dataset_name)
        
        # 修复数据路径
        config['data']['data_path'] = self._find_actual_data_path(dataset_name)
        
        try:
            # 创建数据模块获取实际信息
            data_module = UniversalDataModule(config, for_pretraining=True)
            dataset_info = data_module.get_dataset_info()
            
            return {
                'dataset_name': dataset_name,
                'description': ConfigGenerator.DATASET_CONFIGS[dataset_name]['description'],
                'actual_features': dataset_info['num_features'],
                'expected_features': ConfigGenerator.DATASET_CONFIGS[dataset_name]['expected_features'],
                'seq_len': dataset_info['seq_len'],
                'pred_len': dataset_info['pred_len'],
                'train_size': dataset_info['train_size'],
                'val_size': dataset_info['val_size'],
                'test_size': dataset_info['test_size'],
                'scaler_type': dataset_info['scaler_type'],
                'data_path': dataset_info['data_path']
            }
        except Exception as e:
            self.logger.error(f"获取数据集 {dataset_name} 信息失败: {e}")
            return {
                'dataset_name': dataset_name,
                'error': str(e)
            }

    def run_single_experiment(self, dataset_name: str, epochs: Optional[int] = None, **kwargs) -> Dict:
        """运行单个数据集的实验"""
        self.logger.info(f"开始运行数据集 {dataset_name} 的实验")
        
        try:
            # 生成配置
            config = ConfigGenerator.generate_config(dataset_name, **kwargs)
            
            # 修复数据路径 - 使用实际的目录名匹配
            # 查找实际的数据目录
            actual_data_path = self._find_actual_data_path(dataset_name)
            config['data']['data_path'] = actual_data_path
            
            # 更新epochs
            if epochs:
                config['training']['epochs'] = epochs
            
            # 确保Flow模型存在
            self._ensure_flow_model(config)
            
            # 创建结果目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_dir = os.path.join(self.results_dir, f"{dataset_name}_{timestamp}")
            os.makedirs(experiment_dir, exist_ok=True)
            
            # 保存配置
            config_path = os.path.join(experiment_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            # 导入训练器
            from train import M2MOEPTrainer
            
            # 训练模型
            trainer = M2MOEPTrainer(config)
            trainer.train()
            
            # 评估模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data_module = UniversalDataModule(config)
            
            # 导入evaluate函数
            from evaluate import evaluate_model
            eval_results, predictions, targets = evaluate_model(trainer.model, data_module.get_test_loader(), data_module, device)
            
            # 保存结果
            results = {
                'dataset_name': dataset_name,
                'timestamp': timestamp,
                'config': config,
                'metrics': eval_results,
                'experiment_dir': experiment_dir
            }
            
            results_path = os.path.join(experiment_dir, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"实验完成，结果保存到: {experiment_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"实验失败: {e}")
            return {
                'dataset_name': dataset_name,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }

    def _ensure_flow_model(self, config: Dict):
        """确保Flow模型存在；若不存在则自动预训练"""
        flow_model_path = config['training'].get('flow_model_path', 'flow_model_default.pth')
        if os.path.exists(flow_model_path):
            return  # 已存在
        
        self.logger.info(f"Flow模型未找到，将自动预训练: {flow_model_path}")
        
        try:
            # 导入预训练函数
            from pretrain_flow import pretrain_flow
            pretrain_flow(config)
            self.logger.info("Flow模型预训练完成")
        except Exception as e:
            self.logger.error(f"Flow模型预训练失败: {e}")

    def _find_actual_data_path(self, dataset_name: str) -> str:
        """查找实际的数据路径"""
        # 查找以dataset_name开头的目录
        if os.path.exists(self.base_data_path):
            for item in os.listdir(self.base_data_path):
                if item.startswith(dataset_name):
                    full_path = os.path.join(self.base_data_path, item)
                    if os.path.isdir(full_path):
                        return full_path  # 返回目录路径，而不是文件模式
        
        # 如果找不到，返回默认路径
        return os.path.join(self.base_data_path, dataset_name)

    def run_all_experiments(self, epochs: Optional[int] = None, **kwargs) -> Dict:
        """运行所有可用数据集的实验"""
        self.logger.info("开始运行所有数据集的实验")
        
        available_datasets = self.list_available_datasets()
        results = {}
        
        for dataset_info in available_datasets:
            dataset_name = dataset_info['name']
            
            try:
                result = self.run_single_experiment(dataset_name, epochs=epochs, **kwargs)
                results[dataset_name] = result
            except Exception as e:
                self.logger.error(f"数据集 {dataset_name} 实验失败: {e}")
                results[dataset_name] = {
                    'error': str(e),
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                }
        
        # 保存汇总结果
        summary_path = os.path.join(self.results_dir, f'experiment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"所有实验完成，汇总结果保存到: {summary_path}")
        return results

def main():
    parser = argparse.ArgumentParser(description='通用实验运行器')
    parser.add_argument('--dataset', type=str, help='运行特定数据集的实验')
    parser.add_argument('--all', action='store_true', help='运行所有数据集的实验')
    parser.add_argument('--list', action='store_true', help='列出所有可用数据集')
    parser.add_argument('--summary', type=str, help='获取特定数据集的摘要信息')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--data-path', type=str, default='dataset', help='数据集基础路径')
    
    # 模型参数
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    parser.add_argument('--num-experts', type=int, help='专家数量')
    
    args = parser.parse_args()
    
    # 创建实验运行器
    experiment = UniversalExperiment(base_data_path=args.data_path)
    
    # 构建额外参数
    extra_kwargs = {}
    if args.batch_size:
        extra_kwargs['training'] = {'batch_size': args.batch_size}
    if args.learning_rate:
        if 'training' not in extra_kwargs:
            extra_kwargs['training'] = {}
        extra_kwargs['training']['learning_rate'] = args.learning_rate
    if args.num_experts:
        extra_kwargs['model'] = {'num_experts': args.num_experts}
    
    if args.list:
        experiment.list_available_datasets()
    elif args.summary:
        summary = experiment.get_dataset_summary(args.summary)
        print(json.dumps(summary, indent=2, default=str))
    elif args.all:
        experiment.run_all_experiments(epochs=args.epochs, **extra_kwargs)
    elif args.dataset:
        experiment.run_single_experiment(args.dataset, epochs=args.epochs, **extra_kwargs)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

