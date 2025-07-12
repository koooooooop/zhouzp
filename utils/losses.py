"""
复合损失函数模块 - 完整复原版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging


class CompositeLoss(nn.Module):
    """复合损失函数 - 优化版本"""

    def __init__(self, config):
        super().__init__()
        
        # 添加logger或使用print替代
        self.logger = logging.getLogger(__name__)
        
        # 按照文档的设计，只保留三个核心损失
        loss_weights = config['training'].get('loss_weights', {})
        self.prediction_weight = loss_weights.get('prediction', 1.0)
        self.reconstruction_weight = loss_weights.get('reconstruction', 0.1)
        self.triplet_weight = loss_weights.get('triplet', 0.1)
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss()
        
        # 三元组损失 
        triplet_margin = config['training'].get('triplet_margin', 0.5)
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        
        # 数值稳定性参数 - 优化：预计算常用值
        self.loss_clamp_min = 0.0
        self.loss_clamp_max = 100.0
        self.safe_zero = torch.tensor(0.0)
        self.safe_one = torch.tensor(1.0)
        
        # 性能统计
        self.forward_count = 0
        self.nan_count = 0
        
        print("损失函数:")
        print(f"   - 预测损失权重: {self.prediction_weight}")
        print(f"   - 重构损失权重: {self.reconstruction_weight}")
        print(f"   - 三元组损失权重: {self.triplet_weight}")

    def forward(self, predictions, targets, expert_weights=None, expert_embeddings=None, 
                flow_embeddings=None, flow_log_det=None, **kwargs):
        """
        前向传播 - 优化版本
        """
        self.forward_count += 1
        device = predictions.device
        dtype = predictions.dtype
        
        # 批量数值稳定性检查 - 优化：一次性检查所有张量
        tensors_to_check = [predictions, targets]
        if expert_weights is not None:
            tensors_to_check.append(expert_weights)
        if expert_embeddings is not None:
            tensors_to_check.append(expert_embeddings)
        if flow_embeddings is not None:
            tensors_to_check.append(flow_embeddings)
        
        has_nan_inf = False
        for tensor in tensors_to_check:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                has_nan_inf = True
                break
        
        if has_nan_inf:
            self.nan_count += 1
            print(f"警告: 检测到NaN/Inf值 (第{self.nan_count}次)")
            # 批量修复
            predictions = self._batch_stabilize_tensors([predictions])[0]
            targets = self._batch_stabilize_tensors([targets])[0]
            if expert_weights is not None:
                expert_weights = self._batch_stabilize_tensors([expert_weights])[0]
            if expert_embeddings is not None:
                expert_embeddings = self._batch_stabilize_tensors([expert_embeddings])[0]
            if flow_embeddings is not None:
                flow_embeddings = self._batch_stabilize_tensors([flow_embeddings])[0]
        
        # 预测损失 - 优化：直接使用F.mse_loss
        prediction_loss = F.mse_loss(predictions, targets, reduction='mean')
        prediction_loss = torch.clamp(prediction_loss, min=0.0, max=10.0) 
        
        # 重构损失 - 优化：简化逻辑
        reconstruction_loss = self.safe_zero.to(device).to(dtype)
        if flow_embeddings is not None:
            if flow_embeddings.shape == targets.shape:
                reconstruction_loss = F.mse_loss(flow_embeddings, targets, reduction='mean')
                reconstruction_loss = torch.clamp(reconstruction_loss, min=0.0, max=10.0)
            elif (flow_embeddings.dim() == 3 and targets.dim() == 3 and 
                  flow_embeddings.shape[0] == targets.shape[0] and 
                  flow_embeddings.shape[2] == targets.shape[2] and 
                  flow_embeddings.shape[1] > targets.shape[1]):
                # 提取匹配的时间步
                pred_len = targets.shape[1]
                flow_embeddings_matched = flow_embeddings[:, -pred_len:, :]
                reconstruction_loss = F.mse_loss(flow_embeddings_matched, targets, reduction='mean')
                reconstruction_loss = torch.clamp(reconstruction_loss, min=0.0, max=10.0)
                
        # 三元组损失 - 优化：提前检查条件
        triplet_loss = self.safe_zero.to(device).to(dtype)
        if (expert_weights is not None and expert_embeddings is not None and 
            expert_embeddings.numel() > 0 and expert_weights.numel() > 0):
            try:
                triplet_loss = self.triplet_loss(expert_embeddings, expert_weights)
                triplet_loss = torch.clamp(triplet_loss, min=0.0, max=5.0)
            except Exception as e:
                if self.forward_count % 100 == 0:  # 减少日志频率
                    print(f"警告: 三元组损失计算失败: {e}")
        
        # 总损失计算 - 优化：使用torch.addcmul等高效操作
        total_loss = (self.prediction_weight * prediction_loss + 
                     self.reconstruction_weight * reconstruction_loss +
                     self.triplet_weight * triplet_loss)
        
        # 最终数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("错误: 总损失为NaN或Inf，使用预测损失备份")
            total_loss = prediction_loss
        else:
            total_loss = torch.clamp(total_loss, min=0.0, max=20.0)
        
        return {
            'total': total_loss,
            'prediction': prediction_loss,
            'reconstruction': reconstruction_loss,
            'triplet': triplet_loss
        }

    def _batch_stabilize_tensors(self, tensors):
        """批量数值稳定化处理 - 优化版本"""
        stabilized_tensors = []
        for tensor in tensors:
            if tensor is None:
                stabilized_tensors.append(tensor)
                continue
            
            # 使用torch.nan_to_num一次性处理NaN和Inf
            stabilized = torch.nan_to_num(tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            stabilized = torch.clamp(stabilized, min=-10.0, max=10.0)
            stabilized_tensors.append(stabilized)
        
        return stabilized_tensors

    def _stabilize_tensor(self, tensor, name="tensor"):
        """保留原有接口以保持兼容性"""
        if tensor is None:
            return tensor
        return self._batch_stabilize_tensors([tensor])[0]

    def compute_reconstruction_loss(self, x_original, x_reconstructed):
        """计算重构损失"""
        if x_reconstructed is None:
            return torch.tensor(0.0, device=x_original.device)
        
        try:
            # 确保输入形状一致
            if x_original.dim() == 3:  # [B, seq_len, input_dim]
                x_flat = x_original.view(x_original.size(0), -1)
            else:
                x_flat = x_original
            
            if x_reconstructed.dim() == 3:  # [B, seq_len, input_dim]
                x_recon_flat = x_reconstructed.view(x_reconstructed.size(0), -1)
            else:
                x_recon_flat = x_reconstructed
            
            # 计算重构损失
            recon_loss = F.mse_loss(x_recon_flat, x_flat)
            
            return recon_loss
        except Exception as e:
            # 如果形状不匹配或其他错误，返回0损失
            return torch.tensor(0.0, device=x_original.device)

    def compute_load_balancing_loss(self, routing_weights):
        """计算负载均衡损失"""
        # 计算每个专家的使用率
        expert_usage = routing_weights.mean(dim=0)
        
        # 计算方差：理想情况下所有专家使用率相等
        ideal_usage = 1.0 / routing_weights.size(1)
        balance_loss = torch.var(expert_usage) + F.mse_loss(expert_usage, torch.full_like(expert_usage, ideal_usage))
        
        return balance_loss

    def compute_kl_consistency_loss(self, routing_weights, embeddings=None):
        """KL-一致性损失：对于嵌入空间相近的样本，其路由分布应相似。"""

        batch_size = routing_weights.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=routing_weights.device)

        # 若未提供 embeddings，则直接使用 routing_weights 计算相似度
        if embeddings is None:
            embeddings = routing_weights

        # 计算余弦相似度矩阵
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
        )  # [B,B]

        # 定义"相似"阈值
        similar_mask = (sim_matrix > 0.8).float() - torch.eye(batch_size, device=sim_matrix.device)

        if similar_mask.sum() == 0:
            return torch.tensor(0.0, device=routing_weights.device)

        # 计算 KL(P||Q) + KL(Q||P) 对称 KL
        p = routing_weights.unsqueeze(1)  # [B,1,E]
        q = routing_weights.unsqueeze(0)  # [1,B,E]
        kl_pq = (p * (p.clamp(1e-8).log() - q.clamp(1e-8).log())).sum(-1)
        kl_qp = (q * (q.clamp(1e-8).log() - p.clamp(1e-8).log())).sum(-1)
        sym_kl = kl_pq + kl_qp  # [B,B]

        loss = (sym_kl * similar_mask).sum() / similar_mask.sum()
        return loss

    def update_epoch(self, epoch):
        """更新当前epoch"""
        self.current_epoch = epoch


class TripletLoss(nn.Module):
    """
    按文档设计的三元组损失：基于预测性能构建三元组
    """
    def __init__(self, margin=0.5, mining_strategy='batch_hard'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy

    def forward(self, embeddings, expert_weights):
        """
        按文档设计的三元组损失计算
        文档原理：如果专家k对锚点预测最准确，那么另一个同样由专家k主导的工作负载为正样本
        """
        if embeddings.size(0) < 3:  # 至少需要3个样本构成三元组
            return torch.tensor(0.0, device=embeddings.device)
        
        batch_size = embeddings.size(0)
        
        # === 1. 确定每个样本的主导专家 ===
        dominant_experts = torch.argmax(expert_weights, dim=1)  # [B]
        
        # === 2. 构建正负样本mask ===
        # 正样本：具有相同主导专家的样本
        positive_mask = (dominant_experts.unsqueeze(0) == dominant_experts.unsqueeze(1))  # [B, B]
        # 移除对角线 (自己与自己)
        positive_mask = positive_mask & ~torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        
        # 负样本：具有不同主导专家的样本
        negative_mask = ~positive_mask & ~torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        
        # === 3. 计算嵌入距离矩阵 ===
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # [B, B]
        
        # === 4. Batch Hard Mining (文档推荐策略) ===
        losses = []
        
        for i in range(batch_size):
            # 找到当前锚点的正负样本
            pos_indices = positive_mask[i].nonzero(as_tuple=False).squeeze(1)
            neg_indices = negative_mask[i].nonzero(as_tuple=False).squeeze(1)
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            if self.mining_strategy == 'batch_hard':
                # 最难的正样本：距离最远的正样本
                hardest_positive_dist = dist_matrix[i, pos_indices].max()
                # 最难的负样本：距离最近的负样本
                hardest_negative_dist = dist_matrix[i, neg_indices].min()
                
                # 计算三元组损失
                loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
                losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=embeddings.device)
        
        return torch.stack(losses).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TemperatureScaledCrossEntropy(nn.Module):
    """
    Temperature-scaled cross entropy for calibration
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits, targets):
        scaled_logits = logits / self.temperature
        return F.cross_entropy(scaled_logits, targets)


def compute_expert_usage_entropy(routing_weights):
    """
    计算专家使用熵
    :param routing_weights: 路由权重 [B, E]
    :return: 熵值
    """
    expert_usage = routing_weights.mean(dim=0)
    entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
    return entropy


def compute_expert_diversity_loss(expert_outputs, routing_weights):
    """
    计算专家多样性损失
    :param expert_outputs: 专家输出 [B, E, D]
    :param routing_weights: 路由权重 [B, E]
    :return: 多样性损失
    """
    # 加权专家输出
    weighted_outputs = expert_outputs * routing_weights.unsqueeze(-1)
    
    # 计算专家间的相似度
    similarities = []
    num_experts = expert_outputs.size(1)
    
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            sim = F.cosine_similarity(
                weighted_outputs[:, i, :], 
                weighted_outputs[:, j, :], 
                dim=1
            ).mean()
            similarities.append(sim)
    
    if similarities:
        # 多样性损失：最小化专家间相似度
        diversity_loss = torch.stack(similarities).mean()
    else:
        diversity_loss = torch.tensor(0.0, device=expert_outputs.device)
    
    return diversity_loss


def compute_routing_consistency_loss(routing_weights, embeddings, temperature=0.1):
    """
    计算路由一致性损失
    :param routing_weights: 路由权重 [B, E]
    :param embeddings: 嵌入向量 [B, D]
    :param temperature: 温度参数
    :return: 一致性损失
    """
    # 计算嵌入相似度
    embedding_sim = torch.mm(F.normalize(embeddings), F.normalize(embeddings).t())
    
    # 计算路由权重相似度
    routing_sim = torch.mm(routing_weights, routing_weights.t())
    
    # 一致性损失：嵌入相似的样本应该有相似的路由权重
    consistency_loss = F.mse_loss(
        F.softmax(embedding_sim / temperature, dim=1),
        F.softmax(routing_sim / temperature, dim=1)
    )
    
    return consistency_loss

def calculate_enhanced_load_balancing_loss(routing_weights, num_experts, epsilon=1e-8, diversity_weight=1.0, epoch=0):
    """
    优化的专家负载均衡损失，采用更温和的多样性策略
    """
    batch_size = routing_weights.size(0)
    
    # === 动态权重调整策略 ===
    # 前期(0-10 epoch)：强调多样性，防止早期坍塌
    # 中期(10-30 epoch)：平衡多样性和性能
    # 后期(30+ epoch)：主要关注性能，减少多样性约束
    
    if epoch < 10:
        # 早期：防止专家坍塌
        diversity_factor = 1.0
        balance_factor = 2.0
    elif epoch < 30:
        # 中期：逐渐减少多样性约束
        diversity_factor = max(0.3, 1.0 - (epoch - 10) * 0.035)  # 1.0 -> 0.3
        balance_factor = max(1.0, 2.0 - (epoch - 10) * 0.05)     # 2.0 -> 1.0
    else:
        # 后期：主要关注性能
        diversity_factor = 0.1
        balance_factor = 0.5
    
    # === 专家使用率分析 ===
    expert_usage = routing_weights.mean(dim=0)  # [num_experts]
    max_usage = expert_usage.max()
    min_usage = expert_usage.min()
    usage_ratio = max_usage / (min_usage + epsilon)
    
    # 只有在严重不均衡时才增加惩罚
    collapse_penalty = 1.0
    if usage_ratio > 20.0:  # 提高阈值，更宽松
        collapse_penalty = 1.5 + min(1.0, usage_ratio / 50.0)  # 更温和的惩罚
    
    # === 策略1: 温和的使用率均衡损失 ===
    expert_usage = expert_usage.clamp_min(epsilon)
    target_usage = torch.full_like(expert_usage, 1.0 / num_experts)
    
    # 使用更温和的L2损失替代KL散度
    balance_loss = F.mse_loss(expert_usage, target_usage) * balance_factor * collapse_penalty
    
    # === 策略2: 软多样性约束 ===
    # 只有在多样性因子较高时才计算
    diversity_loss = torch.tensor(0.0, device=routing_weights.device)
    if diversity_factor > 0.2:
        routing_norm = F.normalize(routing_weights.t(), p=2, dim=1)  # [num_experts, batch_size]
        similarity_matrix = torch.mm(routing_norm, routing_norm.t())  # [num_experts, num_experts]
        
        # 移除对角线
        eye_mask = torch.eye(num_experts, device=routing_weights.device)
        off_diagonal_sim = similarity_matrix * (1 - eye_mask)
        
        # 更温和的多样性损失
        diversity_loss = off_diagonal_sim.abs().mean() * diversity_factor
    
    # === 策略3: 自适应熵正则化 ===
    # 目标熵根据训练阶段调整
    routing_entropy = -torch.sum(routing_weights * torch.log(routing_weights.clamp_min(epsilon)), dim=1)
    
    if epoch < 20:
        # 早期：鼓励更高的熵（更分散的路由）
        target_entropy_ratio = 0.8
    else:
        # 后期：允许更集中的路由
        target_entropy_ratio = 0.5
    
    target_entropy = torch.log(torch.tensor(float(num_experts), device=routing_weights.device)) * target_entropy_ratio
    entropy_loss = F.mse_loss(routing_entropy, target_entropy.expand_as(routing_entropy)) * balance_factor * 0.3
    
    # === 组合损失 ===
    total_balance_loss = balance_loss + diversity_loss + entropy_loss
    
    return total_balance_loss