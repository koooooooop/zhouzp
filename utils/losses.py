"""
复合损失函数模块 - 完整复原版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class CompositeLoss(nn.Module):
    """复合损失函数，采用同方差不确定性加权 (Kendall & Gal, 2018)。"""

    def __init__(self, config):
        super().__init__()
        
        # 可学习 log σ 参数 (初值取对数形式)
        loss_weights = config['training'].get('loss_weights', {})
        self.log_sigma_rc = nn.Parameter(torch.log(torch.tensor(loss_weights.get('init_sigma_rc', 1.0))))
        self.log_sigma_cl = nn.Parameter(torch.log(torch.tensor(loss_weights.get('init_sigma_cl', 1.0))))
        self.log_sigma_pr = nn.Parameter(torch.log(torch.tensor(loss_weights.get('init_sigma_pr', 1.0))))
        self.log_sigma_cons = nn.Parameter(torch.log(torch.tensor(loss_weights.get('init_sigma_consistency', 1.0))))
        self.log_sigma_bal = nn.Parameter(torch.log(torch.tensor(loss_weights.get('init_sigma_balance', 1.0))))
        self.log_sigma_recon = nn.Parameter(torch.log(torch.tensor(loss_weights.get('init_sigma_reconstruction', 1.0))))
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # 三元组损失
        triplet_margin = config['training'].get('triplet_margin', 0.5)
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        
        # 当前epoch
        self.current_epoch = 0

    def forward(self, predictions, targets, aux_info, flow_reconstruction=None):
        """
        计算复合损失
        :param predictions: 预测值 [B, pred_len]
        :param targets: 目标值 [B, pred_len]
        :param aux_info: 辅助信息字典，包含路由权重、嵌入等
        :param flow_reconstruction: Flow重构结果（可选）
        """
        # 增强的数值稳定性检查
        stability_issues = []
        
        # 检查预测值
        if torch.isnan(predictions).any():
            nan_count = torch.isnan(predictions).sum().item()
            stability_issues.append(f"预测值包含{nan_count}个NaN")
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isinf(predictions).any():
            inf_count = torch.isinf(predictions).sum().item()
            stability_issues.append(f"预测值包含{inf_count}个Inf")
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 检查数值范围
        if predictions.abs().max() > 1e6:
            max_val = predictions.abs().max().item()
            stability_issues.append(f"预测值过大: {max_val}")
            predictions = torch.clamp(predictions, -1e6, 1e6)
        
        # 检查目标值
        if torch.isnan(targets).any():
            nan_count = torch.isnan(targets).sum().item()
            stability_issues.append(f"目标值包含{nan_count}个NaN")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isinf(targets).any():
            inf_count = torch.isinf(targets).sum().item()
            stability_issues.append(f"目标值包含{inf_count}个Inf")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 检查辅助信息的数值稳定性
        if 'expert_weights' in aux_info:
            weights = aux_info['expert_weights']
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                stability_issues.append("专家权重包含NaN/Inf")
                weights = torch.nan_to_num(weights, nan=1.0/weights.size(-1), posinf=1.0, neginf=0.0)
                # 重新归一化
                weights = F.softmax(weights, dim=-1)
                aux_info['expert_weights'] = weights
        
        # 记录稳定性问题
        if stability_issues:
            print(f"警告: 发现数值稳定性问题: {'; '.join(stability_issues)}")
        
        # 验证修复后的数值
        assert not torch.isnan(predictions).any(), "预测值修复后仍包含NaN"
        assert not torch.isinf(predictions).any(), "预测值修复后仍包含Inf"
        assert not torch.isnan(targets).any(), "目标值修复后仍包含NaN"
        assert not torch.isinf(targets).any(), "目标值修复后仍包含Inf"
        
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        # 1. 预测损失 (MSE + MAE) - 增加数值稳定性检查
        try:
            mse_loss = self.mse_loss(predictions, targets)
            mae_loss = self.mae_loss(predictions, targets)
            
            # 检查损失值的合理性
            if torch.isnan(mse_loss) or torch.isnan(mae_loss):
                print("警告: 预测损失计算出现NaN，使用备用方案")
                mse_loss = torch.tensor(0.0, device=predictions.device)
                mae_loss = torch.tensor(0.0, device=predictions.device)
            
            prediction_loss = mse_loss + 0.1 * mae_loss
            
            # 限制损失值范围
            prediction_loss = torch.clamp(prediction_loss, 0.0, 1e6)
            
        except Exception as e:
            print(f"预测损失计算失败: {e}")
            prediction_loss = torch.tensor(0.0, device=predictions.device)
        
        losses['prediction'] = prediction_loss
        
        # 安全的权重计算
        try:
            sigma_weight = torch.exp(-2 * self.log_sigma_pr)
            if torch.isnan(sigma_weight) or torch.isinf(sigma_weight):
                sigma_weight = torch.tensor(1.0, device=predictions.device)
            total_loss += sigma_weight * prediction_loss + self.log_sigma_pr
        except Exception as e:
            print(f"权重计算失败: {e}")
            total_loss += prediction_loss
        
        # 2. 重构损失 (来自Flow模型) - 增强错误处理
        reconstruction_loss = torch.tensor(0.0, device=predictions.device)
        try:
            if 'reconstruction_loss' in aux_info:
                reconstruction_loss = aux_info['reconstruction_loss']
                if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                    reconstruction_loss = torch.tensor(0.0, device=predictions.device)
            elif flow_reconstruction is not None and 'original_input' in aux_info:
                reconstruction_loss = self.compute_reconstruction_loss(
                    aux_info['original_input'], flow_reconstruction
                )
            
            reconstruction_loss = torch.clamp(reconstruction_loss, 0.0, 1e6)
            
            # 安全的权重计算
            sigma_weight = torch.exp(-2 * self.log_sigma_recon)
            if torch.isnan(sigma_weight) or torch.isinf(sigma_weight):
                sigma_weight = torch.tensor(1.0, device=predictions.device)
            total_loss += sigma_weight * reconstruction_loss + self.log_sigma_recon
            
        except Exception as e:
            print(f"重建损失计算失败: {e}")
            reconstruction_loss = torch.tensor(0.0, device=predictions.device)
        
        losses['reconstruction'] = reconstruction_loss
        
        # 3. 基于预测性能的三元组损失 (来自模型)
        triplet_loss = torch.tensor(0.0, device=predictions.device)
        try:
            if 'triplet_loss' in aux_info:
                triplet_loss = aux_info['triplet_loss']
                if torch.isnan(triplet_loss) or torch.isinf(triplet_loss):
                    triplet_loss = torch.tensor(0.0, device=predictions.device)
                else:
                    triplet_loss = torch.clamp(triplet_loss, 0.0, 1e6)
                    
                # 安全的权重计算
                sigma_weight = torch.exp(-2 * self.log_sigma_cl)
                if torch.isnan(sigma_weight) or torch.isinf(sigma_weight):
                    sigma_weight = torch.tensor(1.0, device=predictions.device)
                total_loss += sigma_weight * triplet_loss + self.log_sigma_cl
        except Exception as e:
            print(f"三元组损失计算失败: {e}")
            triplet_loss = torch.tensor(0.0, device=predictions.device)
        
        losses['triplet'] = triplet_loss
        
        # 4. 一致性损失 - 增强错误处理
        consistency_loss = torch.tensor(0.0, device=predictions.device)
        try:
            if 'expert_weights' in aux_info and 'expert_features' in aux_info:
                consistency_loss = self.compute_kl_consistency_loss(
                    aux_info['expert_weights'], aux_info.get('gating_embeddings', None)
                )
                
                if torch.isnan(consistency_loss) or torch.isinf(consistency_loss):
                    consistency_loss = torch.tensor(0.0, device=predictions.device)
                else:
                    consistency_loss = torch.clamp(consistency_loss, 0.0, 1e6)
                    
                    # 安全的权重计算
                    sigma_weight = torch.exp(-2 * self.log_sigma_cons)
                    if torch.isnan(sigma_weight) or torch.isinf(sigma_weight):
                        sigma_weight = torch.tensor(1.0, device=predictions.device)
                    total_loss += sigma_weight * consistency_loss + self.log_sigma_cons
        except Exception as e:
            print(f"一致性损失计算失败: {e}")
            consistency_loss = torch.tensor(0.0, device=predictions.device)
        
        losses['consistency'] = consistency_loss
        
        # 5. 负载均衡损失 - 增强错误处理
        load_balancing_loss = torch.tensor(0.0, device=predictions.device)
        try:
            if 'load_balance_loss' in aux_info:
                load_balancing_loss = aux_info['load_balance_loss']
                if torch.isnan(load_balancing_loss) or torch.isinf(load_balancing_loss):
                    load_balancing_loss = torch.tensor(0.0, device=predictions.device)
            elif 'expert_weights' in aux_info:
                load_balancing_loss = self.compute_load_balancing_loss(aux_info['expert_weights'])
                
            if not (torch.isnan(load_balancing_loss) or torch.isinf(load_balancing_loss)):
                load_balancing_loss = torch.clamp(load_balancing_loss, 0.0, 1e6)
                
                # 安全的权重计算
                sigma_weight = torch.exp(-2 * self.log_sigma_bal)
                if torch.isnan(sigma_weight) or torch.isinf(sigma_weight):
                    sigma_weight = torch.tensor(1.0, device=predictions.device)
                total_loss += sigma_weight * load_balancing_loss + self.log_sigma_bal
            else:
                load_balancing_loss = torch.tensor(0.0, device=predictions.device)
        except Exception as e:
            print(f"负载均衡损失计算失败: {e}")
            load_balancing_loss = torch.tensor(0.0, device=predictions.device)
        
        losses['load_balance'] = load_balancing_loss
        
        # 6. 专家原型分离损失 - 增强错误处理
        prototype_loss = torch.tensor(0.0, device=predictions.device)
        try:
            if 'prototype_loss' in aux_info:
                prototype_loss = aux_info['prototype_loss']
                if torch.isnan(prototype_loss) or torch.isinf(prototype_loss):
                    prototype_loss = torch.tensor(0.0, device=predictions.device)
                else:
                    prototype_loss = torch.clamp(prototype_loss, 0.0, 1e6)
                    total_loss += prototype_loss * 0.1  # 固定权重
        except Exception as e:
            print(f"原型分离损失计算失败: {e}")
            prototype_loss = torch.tensor(0.0, device=predictions.device)
        
        losses['prototype'] = prototype_loss
        
        # 7. 重建损失 - 使用现有的compute_reconstruction_loss方法
        reconstruction_loss = torch.tensor(0.0, device=predictions.device)
        try:
            if 'reconstruction_loss' in aux_info:
                reconstruction_loss = aux_info['reconstruction_loss']
            else:
                # 如果模型没有提供重建损失，使用现有方法计算
                reconstruction_loss = self.compute_reconstruction_loss(targets, predictions)
            
            if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                reconstruction_loss = torch.tensor(0.0, device=predictions.device)
            else:
                reconstruction_loss = torch.clamp(reconstruction_loss, 0.0, 1e6)
                
                # 安全的权重计算
                sigma_weight = torch.exp(-2 * self.log_sigma_recon)
                if torch.isnan(sigma_weight) or torch.isinf(sigma_weight):
                    sigma_weight = torch.tensor(1.0, device=predictions.device)
                total_loss += sigma_weight * reconstruction_loss + self.log_sigma_recon
        except Exception as e:
            print(f"重建损失计算失败: {e}")
            reconstruction_loss = torch.tensor(0.0, device=predictions.device)
        
        losses['reconstruction'] = reconstruction_loss
        
        # 最终的总损失检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("警告: 总损失包含NaN/Inf，使用预测损失作为备用")
            total_loss = prediction_loss
        
        total_loss = torch.clamp(total_loss, 0.0, 1e6)
        losses['total'] = total_loss
        
        return losses['total'], losses

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
    真正的Triplet Loss for metric learning with Batch Hard Mining
    """
    def __init__(self, margin=0.5, mining_strategy='batch_hard'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy

    def forward(self, embeddings, expert_weights):
        """
        计算triplet loss with batch hard mining
        :param embeddings: 嵌入向量 [B, D]
        :param expert_weights: 专家权重分布 [B, num_experts]
        """
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # 使用专家权重分布确定"相似性"
        # 相似性基于专家权重的余弦相似度
        expert_similarity = F.cosine_similarity(
            expert_weights.unsqueeze(1),    # [B, 1, E]
            expert_weights.unsqueeze(0),    # [1, B, E]
            dim=2
        )  # [B, B]
        
        # 计算嵌入距离矩阵
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # [B, B]
        
        # 创建mask：对角线为False（自己与自己）
        mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()
        
        # 定义正负样本：相似度 > 0.7为正样本，< 0.3为负样本
        positive_mask = (expert_similarity > 0.7) & ~mask
        negative_mask = (expert_similarity < 0.3) & ~mask
        
        losses = []
        
        for i in range(embeddings.size(0)):
            # 找到当前样本的正负样本
            pos_indices = positive_mask[i].nonzero(as_tuple=False).squeeze(1)
            neg_indices = negative_mask[i].nonzero(as_tuple=False).squeeze(1)
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            # Batch Hard Mining
            if self.mining_strategy == 'batch_hard':
                # 最难的正样本：距离最远的正样本
                hardest_positive_dist = dist_matrix[i, pos_indices].max()
                # 最难的负样本：距离最近的负样本
                hardest_negative_dist = dist_matrix[i, neg_indices].min()
            else:
                # 随机采样
                pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
                neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))]
                hardest_positive_dist = dist_matrix[i, pos_idx]
                hardest_negative_dist = dist_matrix[i, neg_idx]
            
            # 计算triplet loss
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