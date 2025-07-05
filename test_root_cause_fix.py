#!/usr/bin/env python3
"""
根因修复验证脚本
测试expert_embeddings的正确生成和梯度稳定性
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

def test_expert_embeddings_fix():
    """测试expert_embeddings缺失问题的修复"""
    print("🔬 测试expert_embeddings缺失问题修复...")
    
    # 加载根因修复配置
    config_path = "configs/weather_ultra_stable.yaml"
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
    
    # 获取测试数据
    train_loader = data_module.get_train_loader()
    batch_x, batch_y = next(iter(train_loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    
    print(f"测试数据形状: {batch_x.shape} -> {batch_y.shape}")
    
    # 测试模型前向传播
    model.eval()
    with torch.no_grad():
        output = model(batch_x, ground_truth=batch_y, return_aux_info=True)
    
    # 检查输出结构
    print("🔍 检查模型输出结构:")
    print(f"   - predictions: {output['predictions'].shape}")
    print(f"   - aux_info keys: {list(output['aux_info'].keys())}")
    
    # 🔧 关键检查：expert_embeddings是否存在
    aux_info = output['aux_info']
    if 'expert_embeddings' in aux_info:
        expert_embeddings = aux_info['expert_embeddings']
        print(f"✅ expert_embeddings存在: {expert_embeddings.shape}")
        
        # 检查expert_embeddings的数值属性
        if torch.isnan(expert_embeddings).any():
            print("❌ expert_embeddings包含NaN")
            return False
        if torch.isinf(expert_embeddings).any():
            print("❌ expert_embeddings包含Inf")
            return False
        
        # 检查数值范围
        emb_min, emb_max = expert_embeddings.min().item(), expert_embeddings.max().item()
        print(f"   - 数值范围: [{emb_min:.4f}, {emb_max:.4f}]")
        
        if abs(emb_min) > 100 or abs(emb_max) > 100:
            print("❌ expert_embeddings数值范围过大")
            return False
        
        print("✅ expert_embeddings数值正常")
        return True
    else:
        print("❌ expert_embeddings仍然缺失")
        return False

def test_flow_stability():
    """测试Flow模型数值稳定性修复"""
    print("\n🔬 测试Flow模型数值稳定性修复...")
    
    # 测试Flow模型的简化版本
    from models.flow import SimpleStableFlow
    
    # 创建测试Flow模型
    input_dim = 16 * 96  # hidden_dim * seq_len
    flow_model = SimpleStableFlow(input_dim, flow_layers=1)
    
    # 测试数据
    batch_size = 4
    test_input = torch.randn(batch_size, input_dim)
    
    print(f"测试Flow模型: 输入形状 {test_input.shape}")
    
    try:
        # 测试前向传播
        z, log_det = flow_model(test_input)
        print(f"✅ Flow前向传播成功: {z.shape}")
        
        # 检查数值稳定性
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("❌ Flow输出包含NaN/Inf")
            return False
        
        if torch.isnan(log_det).any() or torch.isinf(log_det).any():
            print("❌ Flow log_det包含NaN/Inf")
            return False
        
        # 测试重构
        reconstructed = flow_model.reconstruct(test_input)
        print(f"✅ Flow重构成功: {reconstructed.shape}")
        
        # 检查重构质量
        recon_error = torch.mean((test_input - reconstructed) ** 2).item()
        print(f"   - 重构误差: {recon_error:.6f}")
        
        if recon_error > 1.0:
            print("❌ Flow重构误差过大")
            return False
        
        print("✅ Flow模型数值稳定")
        return True
        
    except Exception as e:
        print(f"❌ Flow模型测试失败: {e}")
        return False

def test_gradient_stability_after_fix():
    """测试修复后的梯度稳定性"""
    print("\n🔬 测试修复后的梯度稳定性...")
    
    # 加载根因修复配置
    config_path = "configs/weather_ultra_stable.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化数据模块
    data_module = UniversalDataModule(config)
    
    # 更新配置中的实际特征数
    actual_features = data_module.get_dataset_info()['num_features']
    config['model']['input_dim'] = actual_features
    config['model']['output_dim'] = actual_features
    
    # 初始化模型
    model = M2_MOEP(config).to(device)
    
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
    
    print("📊 开始3个batch的梯度稳定性测试...")
    
    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        if batch_idx >= 3:  # 只测试3个batch
            break
            
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        try:
            # 🔧 关键测试：使用修复后的模型
            output = model(batch_x, ground_truth=batch_y, return_aux_info=True)
            predictions = output['predictions']
            aux_info = output['aux_info']
            
            # 🔧 关键检查：expert_embeddings是否正确传递
            if 'expert_embeddings' not in aux_info:
                print(f"❌ Batch {batch_idx}: expert_embeddings仍然缺失")
                return False
            
            expert_embeddings = aux_info['expert_embeddings']
            print(f"✅ Batch {batch_idx}: expert_embeddings存在 {expert_embeddings.shape}")
            
            # 计算损失
            losses = criterion(
                predictions=predictions,
                targets=batch_y,
                expert_weights=aux_info.get('expert_weights'),
                expert_embeddings=expert_embeddings,  # 🔧 现在应该正确传递
                flow_embeddings=None,
                flow_log_det=aux_info.get('flow_log_det')
            )
            total_loss = losses['total']
            
            # 反向传播
            total_loss.backward()
            
            # 计算梯度范数
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
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
            
            # 🔧 关键检查：梯度范数应该显著减小
            if total_grad_norm > 1.0:  # 比之前的10-80范围大幅减小
                print(f"❌ Batch {batch_idx}: 梯度仍然过大 (norm={total_grad_norm:.4f})")
                return False
            
            optimizer.step()
            
            print(f"✅ Batch {batch_idx}: Loss={total_loss.item():.6f}, GradNorm={total_grad_norm:.6f}")
            
        except Exception as e:
            print(f"❌ Batch {batch_idx}: 训练失败 - {e}")
            return False
    
    # 分析统计结果
    print("\n📈 梯度稳定性分析:")
    print(f"   - 平均梯度范数: {np.mean(gradient_stats):.6f}")
    print(f"   - 最大梯度范数: {np.max(gradient_stats):.6f}")
    print(f"   - 梯度范数标准差: {np.std(gradient_stats):.6f}")
    print(f"   - 平均损失: {np.mean(loss_stats):.6f}")
    print(f"   - 损失标准差: {np.std(loss_stats):.6f}")
    
    # 🔧 关键判断：梯度范数应该在合理范围内
    if np.max(gradient_stats) < 1.0 and np.std(gradient_stats) < 0.5:
        print("✅ 梯度稳定性显著改善！")
        return True
    else:
        print("❌ 梯度仍然不够稳定")
        return False

def main():
    """主测试函数"""
    print("🚀 M²-MOEP根因修复验证测试")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    tests = [
        ("expert_embeddings缺失修复", test_expert_embeddings_fix),
        ("Flow模型数值稳定性", test_flow_stability),
        ("修复后梯度稳定性", test_gradient_stability_after_fix),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*60)
    print("🎯 根因修复测试结果总结:")
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   - {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有根因问题已修复！梯度爆炸问题彻底解决！")
        print("🚀 可以开始正常训练了！")
    else:
        print("⚠️  仍有根因问题需要解决")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 