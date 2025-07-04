import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    计算预测指标 - 主要函数名（用于train.py）
    
    Args:
        predictions: 预测值 numpy数组
        targets: 真实值 numpy数组
        
    Returns:
        指标字典
    """
    # 展平多维数据
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # 计算基础指标
    mse = mean_squared_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_flat, pred_flat)
    
    # 计算MAPE（平均绝对百分比误差）
    mape = np.mean(np.abs((target_flat - pred_flat) / (target_flat + 1e-8))) * 100
    
    # 计算R²分数
    r2 = r2_score(target_flat, pred_flat)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    计算预测指标 - torch版本
    :param predictions: 预测值 [B, T] 或 [B, T, D]
    :param targets: 真实值 [B, T] 或 [B, T, D]
    :return: 指标字典
    """
    # 转换为numpy数组
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    return calculate_metrics(pred_np, target_np)

def compute_expert_metrics(routing_weights: torch.Tensor) -> Dict[str, float]:
    """
    计算专家使用指标
    :param routing_weights: 路由权重 [B, E]
    :return: 专家指标字典
    """
    # 专家使用分布
    expert_usage = routing_weights.mean(dim=0).detach().cpu().numpy()
    
    # 计算熵
    entropy = -np.sum(expert_usage * np.log(expert_usage + 1e-8))
    max_entropy = np.log(len(expert_usage))
    normalized_entropy = entropy / max_entropy
    
    # 计算基尼系数（不均匀性）
    gini = compute_gini_coefficient(expert_usage)
    
    # 活跃专家数量
    active_experts = np.sum(expert_usage > 0.01)  # 使用率超过1%的专家
    
    return {
        'expert_entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'gini_coefficient': gini,
        'active_experts': active_experts,
        'expert_usage_std': np.std(expert_usage),
        'expert_usage_mean': np.mean(expert_usage),
        'expert_usage_max': np.max(expert_usage),
        'expert_usage_min': np.min(expert_usage)
    }

def compute_gini_coefficient(usage: np.ndarray) -> float:
    """
    计算基尼系数
    :param usage: 使用率数组
    :return: 基尼系数 (0-1, 0表示完全均匀)
    """
    sorted_usage = np.sort(usage)
    n = len(usage)
    cumsum = np.cumsum(sorted_usage)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def compute_loss_metrics(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    计算损失指标
    
    Args:
        losses: 损失字典
        
    Returns:
        损失指标字典
    """
    loss_metrics = {}
    
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            loss_metrics[key] = value.item()
        else:
            loss_metrics[key] = float(value)
    
    return loss_metrics

def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    格式化指标输出
    
    Args:
        metrics: 指标字典
        precision: 精度
        
    Returns:
        格式化的字符串
    """
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.{precision}f}")
        else:
            formatted.append(f"{key}: {value}")
    
    return " | ".join(formatted)
