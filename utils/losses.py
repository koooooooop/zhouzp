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
    """复合损失函数 """

    def __init__(self, config):
        super().__init__()
        
        # 添加logger或使用print替代
        self.logger = logging.getLogger(__name__)
        
        # 按照文档的设计，只保留三个核心损失
        loss_weights = config['training'].get('loss_weights', {})
        self.prediction_weight = loss_weights.get('prediction', 1.0)
        self.reconstruction_weight = loss_weights.get('reconstruction', 0.1)
        self.triplet_weight = loss_weights.get('triplet', 0.1)
        
        # 文档设计：移除聚类损失！
        # 移除: self.clustering_weight
        # 移除: self.balance_weight
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss()
        
        # 三元组损失 - 按文档设计
        triplet_margin = config['training'].get('triplet_margin', 0.5)
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        
        # 数值稳定性参数
        self.loss_clamp_min = 0.0
        self.loss_clamp_max = 100.0
        
        print("损失函数:")
        print(f"   - 预测损失权重: {self.prediction_weight}")
        print(f"   - 重构损失权重: {self.reconstruction_weight}")
        print(f"   - 三元组损失权重: {self.triplet_weight}")

    def forward(self, predictions, targets, expert_weights=None, expert_embeddings=None, 
                flow_embeddings=None, flow_log_det=None, **kwargs):
        """
        前向传播
        """
        # 确保所有输入张量都需要梯度
        if not predictions.requires_grad:
            predictions = predictions.requires_grad_(True)
        if not targets.requires_grad:
            targets = targets.requires_grad_(True)
        
        # 输入数值稳定性检查
        predictions = self._stabilize_tensor(predictions, "predictions")
        targets = self._stabilize_tensor(targets, "targets")
        
        # 预测损失 - 🔧 关键修复：使用更保守的损失计算
        prediction_loss = F.mse_loss(predictions, targets, reduction='mean')
        prediction_loss = torch.clamp(prediction_loss, min=0.0, max=10.0)  # 更严格的上限
        
        # 损失数值稳定性检查
        if torch.isnan(prediction_loss) or torch.isinf(prediction_loss):
            print("警告: 预测损失包含NaN或Inf，重置为安全值")
            prediction_loss = torch.tensor(1.0, device=predictions.device, requires_grad=True)
        
        # 重构损失
        reconstruction_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        if flow_embeddings is not None:
            flow_embeddings = self._stabilize_tensor(flow_embeddings, "flow_embeddings")
            if not flow_embeddings.requires_grad:
                flow_embeddings = flow_embeddings.requires_grad_(True)
            
            if flow_embeddings.shape != targets.shape:
                # 如果flow_embeddings的时间维度比targets大，提取最后的pred_len步
                if (flow_embeddings.dim() == 3 and targets.dim() == 3 and 
                    flow_embeddings.shape[0] == targets.shape[0] and 
                    flow_embeddings.shape[2] == targets.shape[2] and 
                    flow_embeddings.shape[1] > targets.shape[1]):
                    
                    # 提取最后的pred_len个时间步
                    pred_len = targets.shape[1]
                    flow_embeddings_matched = flow_embeddings[:, -pred_len:, :]
                    
                    reconstruction_loss = F.mse_loss(flow_embeddings_matched, targets, reduction='mean')
                    reconstruction_loss = torch.clamp(reconstruction_loss, min=0.0, max=10.0)
                    
                    if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                        print("警告: 重构损失包含NaN或Inf，重置为零")
                        reconstruction_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
                else:
                    # 其他维度不匹配情况，跳过重构损失
                    print(f"警告: 重构损失维度不匹配 flow:{flow_embeddings.shape} vs targets:{targets.shape}")
                    reconstruction_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            else:
                reconstruction_loss = F.mse_loss(flow_embeddings, targets, reduction='mean')
                reconstruction_loss = torch.clamp(reconstruction_loss, min=0.0, max=10.0)
                
                if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                    print("警告: 重构损失包含NaN或Inf，重置为零")
                    reconstruction_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # 三元组损失 - 🔧 关键修复：更保守的计算
        triplet_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        if expert_weights is not None and expert_embeddings is not None:
            try:
                # 数值稳定性检查
                expert_weights = self._stabilize_tensor(expert_weights, "expert_weights")
                expert_embeddings = self._stabilize_tensor(expert_embeddings, "expert_embeddings")
                
                # 确保需要梯度
                if not expert_weights.requires_grad:
                    expert_weights = expert_weights.requires_grad_(True)
                if not expert_embeddings.requires_grad:
                    expert_embeddings = expert_embeddings.requires_grad_(True)
                
                # 安全的三元组损失计算
                if expert_embeddings.numel() > 0 and expert_weights.numel() > 0:
                    triplet_loss = self.triplet_loss(expert_embeddings, expert_weights)
                    triplet_loss = torch.clamp(triplet_loss, min=0.0, max=5.0)  # 更严格的上限
                    
                    if torch.isnan(triplet_loss) or torch.isinf(triplet_loss):
                        print("警告: 三元组损失包含NaN或Inf，重置为零")
                        triplet_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
                        
            except Exception as e:
                print(f"警告: 三元组损失计算失败: {e}")
                triplet_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # 总损失计算 - 🔧 关键修复：更保守的权重
        total_loss = (
            self.prediction_weight * prediction_loss +
            self.reconstruction_weight * reconstruction_loss +
            self.triplet_weight * triplet_loss
        )
        
        # 最终数值稳定性检查
        total_loss = torch.clamp(total_loss, min=0.0, max=20.0)  # 更严格的总损失上限
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("错误: 总损失为NaN或Inf，使用预测损失备份")
            total_loss = prediction_loss
        
        # 确保总损失需要梯度
        if not total_loss.requires_grad:
            total_loss = total_loss.requires_grad_(True)
        
        return {
            'total': total_loss,
            'prediction': prediction_loss,
            'reconstruction': reconstruction_loss,
            'triplet': triplet_loss
        }

    def _stabilize_tensor(self, tensor, name="tensor"):
        """数值稳定化处理 - 增强版"""
        if tensor is None:
            return tensor
            
        # 检查和修复NaN
        if torch.isnan(tensor).any():
            print(f"警告: {name}包含NaN，替换为0")
            tensor = torch.nan_to_num(tensor, nan=0.0)
        
        # 检查和修复Inf
        if torch.isinf(tensor).any():
            print(f"警告: {name}包含Inf，进行截断")
            tensor = torch.nan_to_num(tensor, posinf=5.0, neginf=-5.0)
        
        # 数值范围裁剪
        tensor = torch.clamp(tensor, min=-10.0, max=10.0)
        
        return tensor

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