import os
import sys
import torch
import numpy as np

# 将项目根目录加入sys.path，便于脚本独立运行
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.m2_moep import M2_MOEP
from utils.losses import CompositeLoss


def synthetic_config():
    """生成简化配置，用于快速单元测试，不依赖真实数据文件。"""
    return {
        'model': {
            'input_dim': 4,
            'output_dim': 4,
            'hidden_dim': 64,
            'num_experts': 3,
            'seq_len': 16,
            'pred_len': 8,
            'expert_params': {
                'mamba_d_model': 64,
                'mamba_scales': [1, 2]
            },
            'flow': {
                'latent_dim': 32,
                'use_pretrained': False
            },
            'diversity': {
                'prototype_dim': 16,
                'num_prototypes': 6,
                'force_diversity': False
            },
            'temperature': {
                'initial': 1.0
            },
            'triplet': {
                'margin': 0.5
            },
            # 启用 Top-k 稀疏门控
            'top_k': 2
        },
        'training': {
            'loss_weights': {
                'init_sigma_rc': 1.0,
                'init_sigma_cl': 1.0,
                'init_sigma_pr': 1.0,
                'init_sigma_consistency': 1.0,
                'init_sigma_balance': 1.0
            },
            'triplet_margin': 0.5,
            'aux_loss_weight': 0.01,
            'learning_rate': 1e-3
        },
        'data': {
            'seq_len': 16,
            'pred_len': 8
        }
    }


def run_quick_checks():
    """执行快速前向、损失与反向传播测试。"""
    cfg = synthetic_config()
    model = M2_MOEP(cfg)
    model.train()

    batch_size = 4
    x = torch.randn(batch_size, cfg['model']['seq_len'], cfg['model']['input_dim'])
    y = torch.randn(batch_size, cfg['model']['pred_len'], cfg['model']['output_dim'])

    # 前向传播
    out = model(x, ground_truth=y, return_aux_info=True)
    preds = out['predictions']
    aux = out['aux_info']

    assert preds.shape == (batch_size, cfg['model']['pred_len'], cfg['model']['output_dim']), "预测张量形状不匹配"

    # 复合损失
    criterion = CompositeLoss(cfg)
    losses = criterion(preds, y, aux)
    assert torch.isfinite(losses['total']), "总损失出现无穷或 NaN"

    # 反向传播
    losses['total'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm), "梯度包含 NaN/Inf"

    print("✅ Quick unit tests passed. 预测形状: {}  总损失: {:.4f}".format(preds.shape, losses['total'].item()))


if __name__ == "__main__":
    run_quick_checks() 