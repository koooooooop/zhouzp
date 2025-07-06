#!/usr/bin/env python3
"""
M²-MOEP 简单运行脚本
提供预设的数据集配置，类似于iTransformer的使用方式
"""

import subprocess
import sys
import os

# 预设配置
DATASET_CONFIGS = {
    'electricity': {
        'data': 'electricity',
        'root_path': './dataset/electricity/',
        'data_path': 'electricity.csv',
        'seq_len': 96,
        'pred_len': 24,
        'batch_size': 4,
        'learning_rate': 0.00005,
        'num_experts': 6,
        'hidden_dim': 128,
        'train_epochs': 30
    },
    'weather': {
        'data': 'weather',
        'root_path': './dataset/weather/',
        'data_path': 'weather.csv',
        'seq_len': 96,
        'pred_len': 24,
        'batch_size': 8,
        'learning_rate': 0.0001,
        'num_experts': 4,
        'hidden_dim': 64,
        'train_epochs': 20
    },
    'traffic': {
        'data': 'traffic',
        'root_path': './dataset/traffic/',
        'data_path': 'traffic.csv',
        'seq_len': 96,
        'pred_len': 24,
        'batch_size': 4,
        'learning_rate': 0.00005,
        'num_experts': 8,
        'hidden_dim': 128,
        'train_epochs': 25
    },
    'ETTh1': {
        'data': 'ETTh1',
        'root_path': './dataset/ETT/',
        'data_path': 'ETTh1.csv',
        'seq_len': 96,
        'pred_len': 24,
        'batch_size': 16,
        'learning_rate': 0.0001,
        'num_experts': 4,
        'hidden_dim': 64,
        'train_epochs': 15
    },
    'ETTh2': {
        'data': 'ETTh2',
        'root_path': './dataset/ETT/',
        'data_path': 'ETTh2.csv',
        'seq_len': 96,
        'pred_len': 24,
        'batch_size': 16,
        'learning_rate': 0.0001,
        'num_experts': 4,
        'hidden_dim': 64,
        'train_epochs': 15
    },
    'ETTm1': {
        'data': 'ETTm1',
        'root_path': './dataset/ETT/',
        'data_path': 'ETTm1.csv',
        'seq_len': 96,
        'pred_len': 24,
        'batch_size': 16,
        'learning_rate': 0.0001,
        'num_experts': 4,
        'hidden_dim': 64,
        'train_epochs': 15
    },
    'ETTm2': {
        'data': 'ETTm2',
        'root_path': './dataset/ETT/',
        'data_path': 'ETTm2.csv',
        'seq_len': 96,
        'pred_len': 24,
        'batch_size': 16,
        'learning_rate': 0.0001,
        'num_experts': 4,
        'hidden_dim': 64,
        'train_epochs': 15
    }
}

def run_experiment(dataset_name, pred_len=24, model_id=None, **kwargs):
    """运行实验"""
    
    if dataset_name not in DATASET_CONFIGS:
        print(f"错误: 不支持的数据集 '{dataset_name}'")
        print(f"支持的数据集: {', '.join(DATASET_CONFIGS.keys())}")
        return
    
    # 获取预设配置
    config = DATASET_CONFIGS[dataset_name].copy()
    
    # 更新预测长度
    config['pred_len'] = pred_len
    
    # 更新其他参数
    config.update(kwargs)
    
    # 设置模型ID
    if model_id is None:
        model_id = f"M2MOEP_{dataset_name}_pl{pred_len}"
    config['model_id'] = model_id
    
    print(f"🚀 开始运行实验: {model_id}")
    print(f"数据集: {dataset_name}")
    print(f"预测长度: {pred_len}")
    print(f"配置: {config}")
    
    # 构建命令
    cmd = [
        sys.executable, 'train_simple.py',
        '--is_training', '1',
        '--model_id', config['model_id'],
        '--data', config['data'],
        '--root_path', config['root_path'],
        '--data_path', config['data_path'],
        '--seq_len', str(config['seq_len']),
        '--pred_len', str(config['pred_len']),
        '--batch_size', str(config['batch_size']),
        '--learning_rate', str(config['learning_rate']),
        '--num_experts', str(config['num_experts']),
        '--hidden_dim', str(config['hidden_dim']),
        '--train_epochs', str(config['train_epochs'])
    ]
    
    # 添加额外参数
    for key, value in config.items():
        if key not in ['model_id', 'data', 'root_path', 'data_path', 'seq_len', 'pred_len', 
                      'batch_size', 'learning_rate', 'num_experts', 'hidden_dim', 'train_epochs']:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ 实验 {model_id} 完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验 {model_id} 失败: {e}")
        return False

def run_all_datasets(pred_len=24):
    """运行所有数据集的实验"""
    print(f"🎯 开始运行所有数据集的实验 (预测长度: {pred_len})")
    
    results = {}
    for dataset_name in DATASET_CONFIGS.keys():
        print(f"\n{'='*60}")
        success = run_experiment(dataset_name, pred_len=pred_len)
        results[dataset_name] = success
    
    print(f"\n{'='*60}")
    print("📊 实验结果总结:")
    for dataset_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {dataset_name}: {status}")
    
    return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='M²-MOEP 快速运行脚本')
    parser.add_argument('--dataset', type=str, default='electricity',
                        choices=list(DATASET_CONFIGS.keys()) + ['all'],
                        help='数据集名称或all（运行所有数据集）')
    parser.add_argument('--pred_len', type=int, default=24,
                        help='预测长度')
    parser.add_argument('--model_id', type=str, default=None,
                        help='模型ID（可选）')
    parser.add_argument('--list_datasets', action='store_true',
                        help='列出所有支持的数据集')
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("支持的数据集:")
        for name, config in DATASET_CONFIGS.items():
            print(f"  {name}: {config['data_path']}")
        return
    
    if args.dataset == 'all':
        run_all_datasets(pred_len=args.pred_len)
    else:
        run_experiment(args.dataset, pred_len=args.pred_len, model_id=args.model_id)

if __name__ == '__main__':
    main() 