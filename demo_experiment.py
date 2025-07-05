#!/usr/bin/env python3
"""
M²-MOEP 演示实验脚本
快速展示系统功能和实验流程
"""

import os
import sys
import torch
import numpy as np
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.m2_moep import M2_MOEP
from utils.losses import CompositeLoss
from utils.metrics import calculate_metrics
from data.universal_dataset import UniversalDataModule
from configs.config_generator import ConfigGenerator


def demo_experiment():
    """演示实验"""
    
    print("🚀 M²-MOEP 演示实验开始")
    print("=" * 60)
    
    # 检查环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 1. 创建合成数据实验
    print("\n🔬 步骤1: 合成数据实验")
    print("-" * 30)
    
    # 合成数据配置
    synthetic_config = {
        'model': {
            'input_dim': 10,
            'output_dim': 10,
            'hidden_dim': 128,
            'num_experts': 4,
            'seq_len': 24,
            'pred_len': 12,
            'top_k': 3,
            'embedding_dim': 64,
            'expert_params': {
                'mamba_d_model': 128,
                'mamba_scales': [1, 2, 4]
            },
            'flow': {
                'latent_dim': 64,
                'hidden_dim': 128,
                'num_coupling_layers': 4
            }
        },
        'data': {
            'dataset_type': 'synthetic',
            'data_path': 'synthetic',
            'seq_len': 24,
            'pred_len': 12,
            'batch_size': 16,
            'num_workers': 2,
            'synthetic_samples': 1000,
            'noise_level': 0.1
        },
        'training': {
            'epochs': 3,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'loss_weights': {
                'init_sigma_rc': 1.0,
                'init_sigma_cl': 1.0,
                'init_sigma_pr': 1.0,
                'init_sigma_consistency': 1.0,
                'init_sigma_balance': 1.0
            }
        }
    }
    
    try:
        # 创建模型
        model = M2_MOEP(synthetic_config).to(device)
        print(f"✅ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建数据
        data_module = UniversalDataModule(synthetic_config, for_pretraining=True)
        train_loader = data_module.get_train_loader()
        print(f"✅ 数据加载成功，批次数: {len(train_loader)}")
        
        # 创建损失函数
        criterion = CompositeLoss(synthetic_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=synthetic_config['training']['learning_rate'])
        
        # 训练循环
        model.train()
        print("\n🏃 开始训练...")
        
        for epoch in range(synthetic_config['training']['epochs']):
            epoch_losses = []
            epoch_start = time.time()
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                if batch_idx >= 5:  # 限制批次数量
                    break
                
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                output = model(batch_x, ground_truth=batch_y, return_aux_info=True)
                predictions = output['predictions']
                aux_info = output['aux_info']
                
                # 计算损失
                total_loss, losses = criterion(predictions, batch_y, aux_info)
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            epoch_time = time.time() - epoch_start
            avg_loss = np.mean(epoch_losses)
            print(f"   Epoch {epoch+1}/{synthetic_config['training']['epochs']}: "
                  f"损失={avg_loss:.4f}, 时间={epoch_time:.2f}s")
        
        # 评估
        print("\n📊 评估模型...")
        model.eval()
        with torch.no_grad():
            test_batch_x, test_batch_y = next(iter(train_loader))
            test_batch_x = test_batch_x.to(device)
            test_batch_y = test_batch_y.to(device)
            
            output = model(test_batch_x, return_aux_info=True)
            predictions = output['predictions']
            aux_info = output['aux_info']
            
            # 计算指标
            metrics = calculate_metrics(predictions.cpu().numpy(), test_batch_y.cpu().numpy())
            
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   MAE: {metrics['mae']:.4f}")
            print(f"   MAPE: {metrics['mape']:.4f}")
            
            # 专家分析
            expert_weights = aux_info['expert_weights']
            expert_usage = expert_weights.mean(dim=0)
            print(f"   专家使用率: {expert_usage.cpu().numpy()}")
        
        print("✅ 合成数据实验完成！")
        
    except Exception as e:
        print(f"❌ 合成数据实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 2. 真实数据集实验预览
    print("\n🏢 步骤2: 真实数据集实验预览")
    print("-" * 30)
    
    # 列出可用数据集
    datasets = ConfigGenerator.get_supported_datasets()
    print("📋 可用数据集:")
    for i, dataset in enumerate(datasets, 1):
        info = ConfigGenerator.get_dataset_info(dataset)
        print(f"   {i}. {dataset}: {info['description']}")
    
    print("\n💡 运行真实数据集实验的命令:")
    print("   python train.py --config configs/electricity_quick.yaml")
    print("   python universal_experiment.py --dataset electricity --epochs 10")
    print("   python run_all_experiments.py  # 运行所有数据集")
    
    # 3. 系统性能测试
    print("\n⚡ 步骤3: 系统性能测试")
    print("-" * 30)
    
    try:
        # 测试不同批次大小的性能
        batch_sizes = [8, 16, 32, 64]
        print("🔍 批次大小性能测试:")
        
        for batch_size in batch_sizes:
            try:
                # 创建测试数据
                test_input = torch.randn(batch_size, 24, 10).to(device)
                
                # 计时
                start_time = time.time()
                
                with torch.no_grad():
                    output = model(test_input, return_aux_info=True)
                
                end_time = time.time()
                inference_time = end_time - start_time
                throughput = batch_size / inference_time
                
                print(f"   批次大小 {batch_size:2d}: {inference_time*1000:.2f}ms, "
                      f"吞吐量: {throughput:.1f} samples/s")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   批次大小 {batch_size:2d}: GPU内存不足")
                    break
                else:
                    raise e
        
        print("✅ 性能测试完成！")
        
    except Exception as e:
        print(f"❌ 性能测试失败: {str(e)}")
    
    print("\n🎉 演示实验完成！")
    print("=" * 60)
    
    # 4. 下一步建议
    print("\n📝 下一步建议:")
    print("1. 运行完整的真实数据集实验")
    print("2. 调整超参数进行优化")
    print("3. 分析专家网络的特化情况")
    print("4. 比较不同数据集上的性能")
    print("5. 可视化预测结果和专家权重")


if __name__ == "__main__":
    demo_experiment() 