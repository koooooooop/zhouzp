"""
Flow模型预训练脚本
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging

from models.flow import PowerfulNormalizingFlow
from data.universal_dataset import UniversalDataModule


def pretrain_flow_model(config, save_path="flow_model_default.pth"):
    """预训练Flow模型"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    try:
        # 创建数据模块
        data_module = UniversalDataModule(config, for_pretraining=True)
        train_loader = data_module.get_train_loader()
        
        # 计算输入维度
        input_dim = config['model']['input_dim'] * config['model']['seq_len']
        latent_dim = min(512, input_dim // 4)
        
        # 创建Flow模型
        flow_model = PowerfulNormalizingFlow(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=256,
            num_coupling_layers=6
        ).to(device)
        
        # 优化器
        optimizer = optim.Adam(flow_model.parameters(), lr=1e-3)
        
        # 预训练
        flow_model.train()
        total_loss = 0
        num_batches = 0
        
        logger.info("开始Flow模型预训练...")
        logger.info(f"输入维度: {input_dim}, 潜在维度: {latent_dim}")
        
        for epoch in range(3):  # 快速预训练
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(train_loader):
                if batch_idx >= 10:  # 限制批次数量
                    break
                
                # 处理数据解包 - 训练加载器返回 (seq_x, seq_y)
                if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                    batch_x, _ = batch_data  # 只需要输入数据
                else:
                    batch_x = batch_data
                    
                batch_x = batch_x.to(device)
                batch_size = batch_x.size(0)
                
                # 展平输入
                x_flat = batch_x.view(batch_size, -1)
                
                optimizer.zero_grad()
                
                # 前向传播 - 使用forward方法而不是encode
                z, log_det = flow_model.forward(x_flat)
                x_recon = flow_model.inverse(z)
                
                # 重构损失
                recon_loss = nn.MSELoss()(x_recon, x_flat)
                
                # 正则化损失（简化）
                reg_loss = -log_det.mean()
                
                # 总损失
                loss = recon_loss + 0.01 * reg_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / min(10, len(train_loader))
            logger.info(f"Epoch {epoch+1}/3, 平均损失: {avg_loss:.4f}")
        
        # 保存模型
        torch.save({
            'model_state_dict': flow_model.state_dict(),
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'config': config
        }, save_path)
        
        logger.info(f"Flow模型预训练完成，保存至: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Flow模型预训练失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 测试配置
    test_config = {
        'model': {
            'input_dim': 321,
            'seq_len': 96,
            'pred_len': 96
        },
        'data': {
            'data_path': 'dataset/electricity_321个客户的每小时用电量/electricity.csv',
            'seq_len': 96,
            'pred_len': 96,
            'batch_size': 32,
            'num_workers': 2
        }
    }
    
    success = pretrain_flow_model(test_config)
    if success:
        print("Flow模型预训练成功！")
    else:
        print("Flow模型预训练失败！") 