#!/usr/bin/env python3
"""
数据集可用性检查脚本
"""

import os
import sys
from universal_experiment import UniversalExperiment
from configs.config_generator import ConfigGenerator

def check_paper_datasets():
    """检查论文对比数据集的可用性"""
    print("=" * 60)
    print("论文对比数据集可用性检查")
    print("=" * 60)
    
    # 论文中的重合数据集
    paper_datasets = [
        ('weather', 'Weather - 气象站21个气象因子'),
        ('ETTm1', 'ETTm1 - 7个因素的变压器温度变化'),
        ('traffic', 'Traffic - 862个传感器道路占用率'),
        ('electricity', 'Electricity - 321个客户用电量'),
        ('ETTh1', 'ETTh1 - 7个因素的变压器温度变化'),
        ('exchange_rate', 'Exchange - 8个国家汇率变化')
    ]
    
    # 创建实验对象
    experiment = UniversalExperiment(base_data_path='dataset')
    
    # 获取可用数据集
    available_datasets = experiment.list_available_datasets()
    available_names = [ds['name'] for ds in available_datasets]
    
    print(f"检查 {len(paper_datasets)} 个论文对比数据集...\n")
    
    available_count = 0
    for dataset_name, description in paper_datasets:
        if dataset_name in available_names:
            print(f"✅ {dataset_name:<15} - {description}")
            available_count += 1
            
            # 获取数据集详细信息
            try:
                dataset_info = experiment.get_dataset_summary(dataset_name)
                if 'error' not in dataset_info:
                    print(f"   📊 特征数: {dataset_info['actual_features']}")
                    print(f"   📈 样本数: {dataset_info['train_size'] + dataset_info['val_size'] + dataset_info['test_size']}")
                    print(f"   🎯 seq_len: {dataset_info['seq_len']}, pred_len: {dataset_info['pred_len']}")
                else:
                    print(f"   ❌ 信息获取失败: {dataset_info['error']}")
                print()
            except Exception as e:
                print(f"   ❌ 信息获取异常: {e}")
                print()
        else:
            print(f"❌ {dataset_name:<15} - {description} (数据集不可用)")
            print()
    
    print(f"可用数据集: {available_count}/{len(paper_datasets)}")
    
    if available_count > 0:
        print(f"\n🎉 发现 {available_count} 个可用的论文对比数据集！")
        print("可以开始运行论文对比实验")
        return True
    else:
        print("\n❌ 没有找到可用的论文对比数据集")
        print("请检查数据集目录和文件")
        return False

def main():
    """主函数"""
    print("M²-MOEP 数据集检查工具")
    print("目标论文: Non-autoregressive Conditional Diffusion Models for Time Series Prediction")
    print()
    
    # 检查数据集
    datasets_available = check_paper_datasets()
    
    if datasets_available:
        print("\n" + "=" * 60)
        print("运行建议:")
        print("=" * 60)
        print("1. 运行所有对比实验:")
        print("   python paper_comparison_experiment.py")
        print()
        print("2. 快速测试 (少数epochs):")
        print("   python paper_comparison_experiment.py --epochs 10")
        print()
        print("3. 只运行多变量实验:")
        print("   python paper_comparison_experiment.py --modes multivariate")
        print()
        print("4. 运行指定数据集:")
        print("   python paper_comparison_experiment.py --datasets weather traffic")

if __name__ == "__main__":
    main() 