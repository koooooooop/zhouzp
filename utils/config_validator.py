#!/usr/bin/env python3
"""
配置验证器 - 确保所有配置项的完整性
"""

def validate_and_fix_config(config):
    """验证并修复配置"""
    
    # 1. 确保基础结构存在
    if 'data' not in config:
        config['data'] = {}
    if 'model' not in config:
        config['model'] = {}
    if 'training' not in config:
        config['training'] = {}
    
    # 2. 修复data配置
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
    
    # 3. 修复model配置
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
    
    # 4. 计算Flow模型的input_dim
    if 'input_dim' not in config['model']['flow']:
        config['model']['flow']['input_dim'] = config['data']['seq_len'] * config['model']['input_dim']
    
    # 5. 修复training配置
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
    
    # 6. 确保batch_size在两个地方都有
    if 'batch_size' not in config['training']:
        config['training']['batch_size'] = config['data']['batch_size']
    
    return config

def fix_config_compatibility(config):
    """修复配置兼容性问题"""
    # 确保batch_size在正确的位置
    if 'training' in config and 'batch_size' in config['training']:
        if 'data' not in config:
            config['data'] = {}
        if 'batch_size' not in config['data']:
            config['data']['batch_size'] = config['training']['batch_size']
    
    # 确保Flow模型配置正确
    if 'model' in config and 'flow' in config['model']:
        flow_config = config['model']['flow']
        if 'input_dim' not in flow_config:
            # 计算input_dim
            seq_len = config['data'].get('seq_len', 96)
            input_dim = config['model'].get('input_dim', 21)
            flow_config['input_dim'] = seq_len * input_dim
    
    return config