#!/usr/bin/env python3
"""
梯度爆炸修复验证脚本
测试修复后的代码是否能稳定训练
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.m2_moep import M2_MOEP
from data.universal_dataset import UniversalDataModule
from utils.losses import CompositeLoss
from utils.metrics import calculate_metrics
from configs.config_generator import ConfigGenerator

def test_gradient_stability():
    """测试梯度稳定性"""
    print("🔬 开始梯度稳定性测试...")
    
    # 加载超稳定配置
    config_path = "configs/weather_stable.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化数据模块
    data_module = UniversalDataModule(config)
    
    # 更新配置中的实际特征数
    actual_features = data_module.get_dataset_info()['num_features']
    config['model']['input_dim'] = actual_features
    config['model']['output_dim'] = actual_features
    
    # 初始化模型
    model = M2_MOEP(config).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 初始化损失函数
    criterion = CompositeLoss(config)
    
    # 初始化优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )
    
    # 获取训练数据
    train_loader = data_module.get_train_loader()
    
    # 梯度监控
    gradient_stats = []
    loss_stats = []
    
    print("📊 开始5个batch的梯度稳定性测试...")
    
    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        if batch_idx >= 5:  # 只测试5个batch
            break
            
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        try:
            # 模型预测
            output = model(batch_x, ground_truth=batch_y, return_aux_info=True)
            predictions = output['predictions']
            aux_info = output['aux_info']
            
            # 计算损失
            losses = criterion(
                predictions=predictions,
                targets=batch_y,
                expert_weights=aux_info.get('expert_weights'),
                expert_embeddings=aux_info.get('expert_embeddings'),
                flow_embeddings=None,
                flow_log_det=aux_info.get('flow_log_det')
            )
            total_loss = losses['total']
            
            # 反向传播
            total_loss.backward()
            
            # 计算梯度范数
            total_grad_norm = 0.0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    param_count += 1
            total_grad_norm = total_grad_norm ** 0.5
            
            # 记录统计
            gradient_stats.append(total_grad_norm)
            loss_stats.append(total_loss.item())
            
            # 检查数值稳定性
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"❌ Batch {batch_idx}: 损失包含NaN/Inf")
                return False
            
            if torch.isnan(torch.tensor(total_grad_norm)) or torch.isinf(torch.tensor(total_grad_norm)):
                print(f"❌ Batch {batch_idx}: 梯度包含NaN/Inf")
                return False
            
            if total_grad_norm > 100.0:
                print(f"❌ Batch {batch_idx}: 梯度爆炸 (norm={total_grad_norm:.4f})")
                return False
            
            # 应用梯度裁剪
            grad_clip_threshold = config['training']['gradient_clip']
            if total_grad_norm > grad_clip_threshold:
                clip_factor = grad_clip_threshold / (total_grad_norm + 1e-6)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_factor)
                print(f"🔧 Batch {batch_idx}: 梯度裁剪 {total_grad_norm:.4f} -> {grad_clip_threshold:.4f}")
            
            optimizer.step()
            
            print(f"✅ Batch {batch_idx}: Loss={total_loss.item():.4f}, GradNorm={total_grad_norm:.4f}")
            
        except Exception as e:
            print(f"❌ Batch {batch_idx}: 训练失败 - {e}")
            return False
    
    # 分析统计结果
    print("\n📈 梯度稳定性分析:")
    print(f"   - 平均梯度范数: {np.mean(gradient_stats):.4f}")
    print(f"   - 最大梯度范数: {np.max(gradient_stats):.4f}")
    print(f"   - 梯度范数标准差: {np.std(gradient_stats):.4f}")
    print(f"   - 平均损失: {np.mean(loss_stats):.4f}")
    print(f"   - 损失标准差: {np.std(loss_stats):.4f}")
    
    # 判断稳定性
    if np.max(gradient_stats) < 10.0 and np.std(gradient_stats) < 5.0:
        print("✅ 梯度稳定性测试通过！")
        return True
    else:
        print("❌ 梯度仍然不稳定")
        return False

def test_fft_fusion():
    """测试FFT融合的数值稳定性"""
    print("\n🔬 测试FFT融合数值稳定性...")
    
    from models.expert import FFTmsMambaExpert
    
    # 创建测试配置
    config = {
        'model': {
            'input_dim': 64,
            'output_dim': 64,
            'seq_len': 96,
            'pred_len': 96,
            'current_expert_id': 0,
            'expert_params': {
                'mamba_d_model': 32,
                'mamba_scales': [1, 2]
            }
        }
    }
    
    expert = FFTmsMambaExpert(config)
    
    # 测试输入
    batch_size = 4
    seq_len = 96
    input_dim = 64
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    try:
        # 测试FFT融合
        x_proj = expert.input_projection(x)
        fused_x = expert._stable_fft_fusion(x_proj)
        
        # 检查输出
        if torch.isnan(fused_x).any() or torch.isinf(fused_x).any():
            print("❌ FFT融合输出包含NaN/Inf")
            return False
        
        if fused_x.shape != x_proj.shape:
            print(f"❌ FFT融合维度不匹配: {fused_x.shape} vs {x_proj.shape}")
            return False
        
        print("✅ FFT融合数值稳定性测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ FFT融合测试失败: {e}")
        return False

def test_loss_function():
    """测试损失函数的数值稳定性"""
    print("\n🔬 测试损失函数数值稳定性...")
    
    config = {
        'training': {
            'loss_weights': {
                'prediction': 1.0,
                'reconstruction': 0.01,
                'triplet': 0.01
            },
            'triplet_margin': 0.1
        }
    }
    
    criterion = CompositeLoss(config)
    
    # 测试数据
    batch_size = 8
    pred_len = 96
    output_dim = 21
    
    predictions = torch.randn(batch_size, pred_len, output_dim)
    targets = torch.randn(batch_size, pred_len, output_dim)
    expert_weights = torch.softmax(torch.randn(batch_size, 2), dim=1)
    expert_embeddings = torch.randn(batch_size, pred_len, 128)
    
    try:
        losses = criterion(
            predictions=predictions,
            targets=targets,
            expert_weights=expert_weights,
            expert_embeddings=expert_embeddings
        )
        
        total_loss = losses['total']
        
        # 检查损失
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("❌ 损失函数输出包含NaN/Inf")
            return False
        
        if total_loss.item() > 100.0:
            print(f"❌ 损失值过大: {total_loss.item()}")
            return False
        
        print(f"✅ 损失函数测试通过！总损失: {total_loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 M²-MOEP梯度爆炸修复验证测试")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    tests = [
        ("FFT融合稳定性", test_fft_fusion),
        ("损失函数稳定性", test_loss_function),
        ("梯度稳定性", test_gradient_stability),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*50)
    print("🎯 测试结果总结:")
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   - {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！梯度爆炸问题已修复！")
    else:
        print("⚠️  仍有问题需要解决")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 