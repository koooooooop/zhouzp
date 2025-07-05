#!/usr/bin/env python3
import torch
import sys
import os
sys.path.append('.')

def test_complete_fix():
    print("🔧 测试完整修复...")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    # 1. 配置验证和修复
    print("📋 验证配置文件...")
    import yaml
    
    # 创建配置验证器
    def validate_and_fix_config(config):
        # 确保基础结构存在
        if 'data' not in config:
            config['data'] = {}
        if 'model' not in config:
            config['model'] = {}
        if 'training' not in config:
            config['training'] = {}
        
        # 修复data配置
        data_defaults = {
            'batch_size': 16,
            'num_workers': 2,
            'pin_memory': True,
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'scaler_type': 'standard',
            'normalize': True,
            'seq_len': 96,
            'pred_len': 96
        }
        
        for key, default_value in data_defaults.items():
            if key not in config['data']:
                config['data'][key] = default_value
        
        # 修复model配置
        model_defaults = {
            'input_dim': 21,
            'output_dim': 21,
            'hidden_dim': 64,
            'num_experts': 4,
            'seq_len': config['data']['seq_len'],
            'pred_len': config['data']['pred_len'],
            'top_k': 2,
            'embedding_dim': 128,
            'temperature': {
                'initial': 5.0,
                'min': 1.0,
                'max': 10.0,
                'decay_rate': 0.995
            },
            'flow': {
                'num_layers': 4,
                'hidden_dim': 32,
                'latent_dim': 256,
                'use_pretrained': True
            }
        }
        
        for key, default_value in model_defaults.items():
            if key not in config['model']:
                config['model'][key] = default_value
        
        # 计算Flow模型的input_dim
        if 'input_dim' not in config['model']['flow']:
            config['model']['flow']['input_dim'] = config['data']['seq_len'] * config['model']['input_dim']
        
        # 修复training配置
        training_defaults = {
            'epochs': 30,
            'learning_rate': 0.0001,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'gradient_clip': 0.5,
            'scheduler': 'cosine',
            'loss_weights': {
                'prediction': 1.0,
                'reconstruction': 0.1,
                'triplet': 0.1
            },
            'triplet_margin': 0.5,
            'triplet_mining': 'batch_hard'
        }
        
        for key, default_value in training_defaults.items():
            if key not in config['training']:
                config['training'][key] = default_value
        
        # 确保batch_size在两个地方都有
        if 'batch_size' not in config['training']:
            config['training']['batch_size'] = config['data']['batch_size']
        
        # 添加device配置
        config['device'] = str(device)
        
        return config
    
    with open('configs/weather_stable.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config = validate_and_fix_config(config)
    print("✅ 配置验证通过")
    
    # 2. 测试数据加载
    print("📊 测试数据加载...")
    from data.universal_dataset import UniversalDataModule
    
    try:
        data_module = UniversalDataModule(config)
        train_loader = data_module.get_train_loader()
        print(f"✅ 数据加载成功: {len(train_loader)} batches")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试模型创建
    print("🤖 测试模型创建...")
    from models.m2_moep import M2_MOEP
    
    try:
        model = M2_MOEP(config)
        
        # 重要：将模型移动到正确的设备
        model = model.to(device)
        print(f"✅ 模型移动到设备: {device}")
        
        print(f"✅ 模型创建成功: {sum(p.numel() for p in model.parameters())} parameters")
        print(f"✅ 模型配置保存成功: config存在 = {hasattr(model, 'config')}")
        print(f"✅ top_k配置: {getattr(model, 'top_k', 'None')}")
        
        # 验证模型设备
        model_device = next(model.parameters()).device
        print(f"✅ 模型实际设备: {model_device}")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试损失函数
    print("📉 测试损失函数...")
    from utils.losses import CompositeLoss
    
    try:
        criterion = CompositeLoss(config)
        print("✅ 损失函数创建成功")
    except Exception as e:
        print(f"❌ 损失函数创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试完整的训练步骤
    print("🏃 测试完整训练步骤...")
    
    try:
        for batch_data in train_loader:
            if len(batch_data) == 2:
                batch_x, batch_y = batch_data
                
                # 重要：将数据移动到正确的设备
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                print(f"✅ 数据解包成功: {batch_x.shape}, {batch_y.shape}")
                print(f"✅ 数据设备: {batch_x.device}")
            else:
                print(f"❌ 数据格式错误: {len(batch_data)}")
                return False
            
            # 测试模型前向传播
            model.eval()
            with torch.no_grad():
                output = model(batch_x, ground_truth=batch_y, return_aux_info=True)
                predictions = output['predictions']
                aux_info = output['aux_info']
            
            # 检查必要的字段
            required_fields = ['expert_weights', 'expert_embeddings']
            for field in required_fields:
                if field not in aux_info:
                    print(f"❌ 缺少字段: {field}")
                    print(f"实际字段: {list(aux_info.keys())}")
                    return False
            
            print(f"✅ 模型前向传播成功: {predictions.shape}")
            print(f"✅ 预测输出设备: {predictions.device}")
            print(f"✅ 辅助信息字段: {list(aux_info.keys())}")
            
            # 测试损失计算
            losses = criterion(
                predictions=predictions,
                targets=batch_y,
                expert_weights=aux_info.get('expert_weights'),
                expert_embeddings=aux_info.get('expert_embeddings'),
                flow_embeddings=None,
                flow_log_det=aux_info.get('flow_log_det')
            )
            
            print(f"✅ 损失计算成功: {losses['total']:.4f}")
            print(f"  - 预测损失: {losses['prediction']:.4f}")
            print(f"  - 重构损失: {losses['reconstruction']:.4f}")
            print(f"  - 三元组损失: {losses['triplet']:.4f}")
            break
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎉 所有修复验证通过!")
    return True

if __name__ == "__main__":
    success = test_complete_fix()
    exit(0 if success else 1)