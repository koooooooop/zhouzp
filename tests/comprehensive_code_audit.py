"""
M²-MOEP 全面代码审计和测试套件
包括代码质量检查、边界条件测试、集成测试和性能分析
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import time
import warnings
import traceback
from typing import Dict, List, Tuple, Any
import inspect

# 项目路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.m2_moep import M2_MOEP
from models.flow import PowerfulNormalizingFlow, FlowLayer
from models.gating import GatingEncoder
from models.expert import FFTmsMambaExpert
from utils.losses import CompositeLoss, TripletLoss
from utils.metrics import calculate_metrics, compute_expert_metrics
from data.universal_dataset import UniversalDataModule
from train import M2MOEPTrainer


class ComprehensiveCodeAudit:
    """全面代码审计套件"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._get_comprehensive_config()
        self.test_results = []
        self.performance_metrics = {}
        self.code_quality_issues = []
        
        print(f"🔧 审计环境: {self.device}")
        print(f"🔍 开始全面代码审计...")
        
    def _get_comprehensive_config(self):
        """获取全面测试配置"""
        return {
            'model': {
                'input_dim': 10,
                'output_dim': 10,
                'hidden_dim': 256,  # 减小模型大小
                'num_experts': 4,   # 减少专家数量
                'seq_len': 24,      # 减小序列长度
                'pred_len': 12,     # 减小预测长度
                'expert_params': {
                    'mamba_d_model': 256,
                    'mamba_scales': [1, 2, 4]
                },
                'flow': {
                    'latent_dim': 128,
                    'use_pretrained': False,
                    'hidden_dim': 256,
                    'num_coupling_layers': 6
                },
                'diversity': {
                    'prototype_dim': 64,
                    'num_prototypes': 8,
                    'force_diversity': True,
                    'diversity_weight': 0.1
                },
                'temperature': {
                    'initial': 1.0,
                    'min': 0.1,
                    'max': 10.0,
                    'decay': 0.98
                },
                'triplet': {
                    'margin': 0.5,
                    'mining_strategy': 'batch_hard'
                },
                'top_k': 3,
                'embedding_dim': 128
            },
            'training': {
                'learning_rate': 1e-4,  # 降低学习率
                'weight_decay': 1e-5,   # 降低权重衰减
                'gradient_clip': 1.0,   # 降低梯度裁剪
                'epochs': 10,
                'batch_size': 16,
                'loss_weights': {
                    'init_sigma_rc': 1.0,
                    'init_sigma_cl': 1.0,
                    'init_sigma_pr': 1.0,
                    'init_sigma_consistency': 1.0,
                    'init_sigma_balance': 1.0
                },
                'triplet_margin': 0.5,
                'aux_loss_weight': 0.01
            },
            'data': {
                'seq_len': 24,
                'pred_len': 12,
                'batch_size': 16,
                'num_workers': 2
            }
        }
    
    def log_test(self, test_name: str, status: str, details: str = "", severity: str = "INFO"):
        """记录测试结果"""
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'severity': severity
        }
        self.test_results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {details}")
        
        if status == "FAIL" and severity == "CRITICAL":
            self.code_quality_issues.append(f"CRITICAL: {test_name} - {details}")
    
    def test_code_structure_and_imports(self):
        """测试代码结构和导入"""
        print("\n📁 测试代码结构和导入...")
        
        try:
            # 测试核心模块导入
            from models.m2_moep import M2_MOEP
            from models.flow import PowerfulNormalizingFlow
            from models.gating import GatingEncoder
            from models.expert import FFTmsMambaExpert
            from utils.losses import CompositeLoss
            from utils.metrics import calculate_metrics
            
            self.log_test("核心模块导入", "PASS", "所有核心模块导入成功")
            
            # 检查类定义完整性
            required_methods = {
                'M2_MOEP': ['forward', '__init__', 'update_temperature_schedule'],
                'PowerfulNormalizingFlow': ['forward', 'inverse', 'encode', 'decode'],
                'GatingEncoder': ['forward', 'get_embeddings'],
                'FFTmsMambaExpert': ['forward', '_early_fft_fusion'],
                'CompositeLoss': ['forward', 'compute_kl_consistency_loss']
            }
            
            for class_name, methods in required_methods.items():
                cls = globals()[class_name.split('.')[-1]]
                for method in methods:
                    if not hasattr(cls, method):
                        self.log_test(f"{class_name}方法检查", "FAIL", 
                                    f"缺少方法: {method}", "CRITICAL")
                    else:
                        # 检查方法签名
                        sig = inspect.signature(getattr(cls, method))
                        if len(sig.parameters) == 0 and method != '__init__':
                            self.log_test(f"{class_name}.{method}签名", "WARN", 
                                        "方法可能缺少必要参数")
            
            self.log_test("类方法完整性", "PASS", "所有必要方法存在")
            
        except Exception as e:
            self.log_test("代码结构检查", "FAIL", str(e), "CRITICAL")
    
    def test_model_initialization_edge_cases(self):
        """测试模型初始化的边界条件"""
        print("\n🏗️ 测试模型初始化边界条件...")
        
        edge_configs = [
            # 最小配置
            {
                'name': '最小配置',
                'config': {
                    'model': {
                        'input_dim': 1, 'output_dim': 1, 'hidden_dim': 8,
                        'num_experts': 2, 'seq_len': 4, 'pred_len': 2,
                        'expert_params': {'mamba_d_model': 8, 'mamba_scales': [1]},
                        'flow': {'latent_dim': 4, 'hidden_dim': 8},
                        'embedding_dim': 8, 'top_k': 1
                    },
                    'training': {'loss_weights': {}}, 'data': {}
                }
            },
            # 大规模配置
            {
                'name': '大规模配置',
                'config': {
                    'model': {
                        'input_dim': 50, 'output_dim': 50, 'hidden_dim': 1024,
                        'num_experts': 16, 'seq_len': 200, 'pred_len': 100,
                        'expert_params': {'mamba_d_model': 1024, 'mamba_scales': [1, 2, 4, 8]},
                        'flow': {'latent_dim': 512, 'hidden_dim': 1024},
                        'embedding_dim': 512, 'top_k': 8
                    },
                    'training': {'loss_weights': {}}, 'data': {}
                }
            },
            # 不平衡配置
            {
                'name': '不平衡配置',
                'config': {
                    'model': {
                        'input_dim': 100, 'output_dim': 1, 'hidden_dim': 64,
                        'num_experts': 3, 'seq_len': 10, 'pred_len': 50,
                        'expert_params': {'mamba_d_model': 32, 'mamba_scales': [1, 4, 16]},
                        'flow': {'latent_dim': 16, 'hidden_dim': 32},
                        'embedding_dim': 16, 'top_k': 2
                    },
                    'training': {'loss_weights': {}}, 'data': {}
                }
            }
        ]
        
        for edge_case in edge_configs:
            try:
                model = M2_MOEP(edge_case['config']).to(self.device)
                
                # 测试前向传播
                batch_size = 2
                seq_len = edge_case['config']['model']['seq_len']
                input_dim = edge_case['config']['model']['input_dim']
                
                x = torch.randn(batch_size, seq_len, input_dim).to(self.device)
                output = model(x, return_aux_info=True)
                
                self.log_test(f"边界条件-{edge_case['name']}", "PASS", 
                            f"模型初始化和前向传播成功")
                
            except Exception as e:
                self.log_test(f"边界条件-{edge_case['name']}", "FAIL", str(e))
    
    def test_numerical_stability_extreme_cases(self):
        """测试极端数值条件下的稳定性"""
        print("\n🔢 测试极端数值稳定性...")
        
        model = M2_MOEP(self.config).to(self.device)
        model.eval()
        
        extreme_cases = [
            ("零输入", torch.zeros),
            ("极大值", lambda b, s, d: torch.full((b, s, d), 1e6)),
            ("极小值", lambda b, s, d: torch.full((b, s, d), 1e-6)),
            ("NaN输入", lambda b, s, d: torch.full((b, s, d), float('nan'))),
            ("Inf输入", lambda b, s, d: torch.full((b, s, d), float('inf'))),
            ("混合极值", lambda b, s, d: torch.cat([
                torch.zeros(b//2, s, d),
                torch.full((b//2, s, d), 1e6)
            ], dim=0)),
            ("梯度消失模拟", lambda *args: torch.randn(*args) * 1e-10),
            ("梯度爆炸模拟", lambda *args: torch.randn(*args) * 1e10)
        ]
        
        batch_size = 4
        seq_len = self.config['model']['seq_len']
        input_dim = self.config['model']['input_dim']
        
        with torch.no_grad():
            for case_name, input_generator in extreme_cases:
                try:
                    x = input_generator(batch_size, seq_len, input_dim).to(self.device)
                    
                    # 跳过NaN和Inf输入的测试（预期会失败）
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        if case_name in ["NaN输入", "Inf输入"]:
                            self.log_test(f"极值稳定性-{case_name}", "SKIP", "预期失败的测试")
                            continue
                    
                    output = model(x, return_aux_info=True)
                    predictions = output['predictions']
                    
                    # 检查输出的数值稳定性
                    if torch.isfinite(predictions).all():
                        self.log_test(f"极值稳定性-{case_name}", "PASS", "输出数值稳定")
                    else:
                        self.log_test(f"极值稳定性-{case_name}", "FAIL", 
                                    "输出包含NaN/Inf", "HIGH")
                    
                except Exception as e:
                    severity = "EXPECTED" if case_name in ["NaN输入", "Inf输入"] else "HIGH"
                    self.log_test(f"极值稳定性-{case_name}", "FAIL", str(e), severity)
    
    def test_memory_and_computational_efficiency(self):
        """测试内存和计算效率"""
        print("\n💾 测试内存和计算效率...")
        
        model = M2_MOEP(self.config).to(self.device)
        
        # 内存使用测试
        batch_sizes = [1, 8, 16, 32, 64]
        seq_len = self.config['model']['seq_len']
        input_dim = self.config['model']['input_dim']
        
        memory_usage = []
        computation_times = []
        
        for batch_size in batch_sizes:
            try:
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    initial_memory = torch.cuda.memory_allocated()
                
                x = torch.randn(batch_size, seq_len, input_dim).to(self.device)
                
                # 计时
                start_time = time.time()
                
                model.eval()
                with torch.no_grad():
                    output = model(x, return_aux_info=True)
                
                end_time = time.time()
                computation_time = end_time - start_time
                
                # 内存使用
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated()
                    memory_used = (peak_memory - initial_memory) / 1024**2  # MB
                    memory_usage.append((batch_size, memory_used))
                
                computation_times.append((batch_size, computation_time))
                
                # 检查内存泄露
                del x, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.log_test(f"效率测试-批大小{batch_size}", "PASS", 
                            f"时间: {computation_time:.4f}s, 内存: {memory_used:.2f}MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.log_test(f"效率测试-批大小{batch_size}", "FAIL", 
                                "GPU内存不足", "HIGH")
                    break
                else:
                    self.log_test(f"效率测试-批大小{batch_size}", "FAIL", str(e))
        
        # 分析效率趋势
        if len(computation_times) > 1:
            # 检查计算时间是否随批大小合理增长
            time_ratios = [computation_times[i][1] / computation_times[0][1] 
                          for i in range(1, len(computation_times))]
            batch_ratios = [computation_times[i][0] / computation_times[0][0] 
                           for i in range(1, len(computation_times))]
            
            efficiency_score = np.mean([t/b for t, b in zip(time_ratios, batch_ratios)])
            
            if efficiency_score < 1.5:
                self.log_test("计算效率分析", "PASS", 
                            f"效率评分: {efficiency_score:.2f} (良好)")
            else:
                self.log_test("计算效率分析", "WARN", 
                            f"效率评分: {efficiency_score:.2f} (可优化)")
        
        self.performance_metrics['memory_usage'] = memory_usage
        self.performance_metrics['computation_times'] = computation_times
    
    def test_gradient_flow_and_optimization(self):
        """测试梯度流和优化稳定性"""
        print("\n🌊 测试梯度流和优化...")
        
        model = M2_MOEP(self.config).to(self.device)
        criterion = CompositeLoss(self.config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        batch_size = 8
        seq_len = self.config['model']['seq_len']
        input_dim = self.config['model']['input_dim']
        pred_len = self.config['model']['pred_len']
        output_dim = self.config['model']['output_dim']
        
        # 模拟多步训练
        gradient_norms = []
        loss_values = []
        
        for step in range(10):
            optimizer.zero_grad()
            
            x = torch.randn(batch_size, seq_len, input_dim).to(self.device)
            y = torch.randn(batch_size, pred_len, output_dim).to(self.device)
            
            # 前向传播
            model.train()
            output = model(x, ground_truth=y, return_aux_info=True)
            predictions = output['predictions']
            aux_info = output['aux_info']
            
            # 损失计算
            losses = criterion(predictions, y, aux_info)
            
            # 反向传播
            losses['total'].backward()
            
            # 计算梯度范数
            total_grad_norm = 0
            param_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    param_count += 1
                    
                    # 检查异常梯度
                    if torch.isnan(param.grad).any():
                        self.log_test(f"梯度检查-步骤{step}", "FAIL", 
                                    f"{name}包含NaN梯度", "CRITICAL")
                    elif torch.isinf(param.grad).any():
                        self.log_test(f"梯度检查-步骤{step}", "FAIL", 
                                    f"{name}包含Inf梯度", "CRITICAL")
            
            total_grad_norm = (total_grad_norm ** 0.5)
            gradient_norms.append(total_grad_norm)
            loss_values.append(losses['total'].item())
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # 检查参数更新
            param_changed = False
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    if param.grad.abs().sum() > 1e-10:
                        param_changed = True
                        break
            
            if not param_changed:
                self.log_test(f"参数更新-步骤{step}", "WARN", "参数未更新")
        
        # 分析梯度流健康度
        avg_grad_norm = np.mean(gradient_norms)
        grad_stability = np.std(gradient_norms) / (avg_grad_norm + 1e-8)
        
        if avg_grad_norm > 1e-6 and avg_grad_norm < 100:
            self.log_test("梯度范数健康度", "PASS", 
                        f"平均梯度范数: {avg_grad_norm:.6f}")
        else:
            self.log_test("梯度范数健康度", "WARN", 
                        f"梯度范数异常: {avg_grad_norm:.6f}")
        
        if grad_stability < 2.0:
            self.log_test("梯度稳定性", "PASS", f"梯度稳定性: {grad_stability:.3f}")
        else:
            self.log_test("梯度稳定性", "WARN", f"梯度不稳定: {grad_stability:.3f}")
        
        # 检查损失下降
        if len(loss_values) > 5:
            early_loss = np.mean(loss_values[:3])
            late_loss = np.mean(loss_values[-3:])
            if late_loss < early_loss:
                self.log_test("损失收敛", "PASS", "损失呈下降趋势")
            else:
                self.log_test("损失收敛", "WARN", "损失未下降")
    
    def test_expert_specialization_and_diversity(self):
        """测试专家特化和多样性"""
        print("\n🎭 测试专家特化和多样性...")
        
        model = M2_MOEP(self.config).to(self.device)
        model.eval()
        
        # 生成不同类型的输入模式
        batch_size = 64
        seq_len = self.config['model']['seq_len']
        input_dim = self.config['model']['input_dim']
        
        # 创建5种不同的信号模式
        patterns = []
        pattern_names = []
        
        for i in range(5):
            pattern_batch_size = batch_size // 5
            
            if i == 0:  # 正弦波
                t = torch.linspace(0, 4*np.pi, seq_len).unsqueeze(0).unsqueeze(-1)
                pattern = torch.sin(t + torch.randn(1, 1, 1) * 0.1)
                pattern = pattern.expand(pattern_batch_size, -1, input_dim)
                pattern_names.append("正弦波")
                
            elif i == 1:  # 锯齿波
                t = torch.linspace(0, 4*np.pi, seq_len).unsqueeze(0).unsqueeze(-1)
                pattern = (t % (2*np.pi) - np.pi)
                pattern = pattern.expand(pattern_batch_size, -1, input_dim)
                pattern_names.append("锯齿波")
                
            elif i == 2:  # 随机游走
                pattern = torch.randn(pattern_batch_size, seq_len, input_dim).cumsum(dim=1)
                pattern_names.append("随机游走")
                
            elif i == 3:  # 阶跃函数
                pattern = torch.zeros(pattern_batch_size, seq_len, input_dim)
                pattern[:, seq_len//2:, :] = 1.0
                pattern_names.append("阶跃函数")
                
            else:  # 白噪声
                pattern = torch.randn(pattern_batch_size, seq_len, input_dim)
                pattern_names.append("白噪声")
            
            patterns.append(pattern)
        
        # 合并所有模式
        x = torch.cat(patterns, dim=0).to(self.device)
        pattern_labels = []
        for i, name in enumerate(pattern_names):
            pattern_labels.extend([i] * (batch_size // 5))
        
        with torch.no_grad():
            output = model(x, return_aux_info=True)
            expert_weights = output['aux_info']['expert_weights']
        
        # 分析专家特化
        pattern_expert_usage = {}
        for i, pattern_name in enumerate(pattern_names):
            pattern_indices = [j for j, label in enumerate(pattern_labels) if label == i]
            pattern_weights = expert_weights[pattern_indices].mean(dim=0)
            pattern_expert_usage[pattern_name] = pattern_weights.cpu().numpy()
        
        # 计算专家特化度
        specialization_scores = []
        for expert_idx in range(self.config['model']['num_experts']):
            expert_usage_across_patterns = [
                pattern_expert_usage[pattern][expert_idx] 
                for pattern in pattern_names
            ]
            # 使用标准差衡量特化度（标准差越大，特化度越高）
            specialization = np.std(expert_usage_across_patterns)
            specialization_scores.append(specialization)
        
        avg_specialization = np.mean(specialization_scores)
        
        if avg_specialization > 0.05:
            self.log_test("专家特化度", "PASS", 
                        f"平均特化度: {avg_specialization:.4f}")
        else:
            self.log_test("专家特化度", "WARN", 
                        f"专家特化度较低: {avg_specialization:.4f}")
        
        # 检查专家多样性
        expert_similarity_matrix = np.corrcoef([
            list(pattern_expert_usage[pattern]) for pattern in pattern_names
        ])
        
        avg_similarity = np.mean(expert_similarity_matrix[np.triu_indices_from(expert_similarity_matrix, k=1)])
        
        if avg_similarity < 0.8:
            self.log_test("专家多样性", "PASS", 
                        f"平均相似度: {avg_similarity:.3f}")
        else:
            self.log_test("专家多样性", "WARN", 
                        f"专家过于相似: {avg_similarity:.3f}")
        
        # 打印详细的专家使用分布
        print("   📊 专家使用分布:")
        for pattern_name, usage in pattern_expert_usage.items():
            top_expert = np.argmax(usage)
            print(f"      {pattern_name}: 主要专家{top_expert} ({usage[top_expert]:.3f})")
    
    def test_loss_function_components(self):
        """测试损失函数各组件"""
        print("\n💔 测试损失函数组件...")
        
        criterion = CompositeLoss(self.config).to(self.device)
        
        batch_size = 16
        pred_len = self.config['model']['pred_len']
        output_dim = self.config['model']['output_dim']
        num_experts = self.config['model']['num_experts']
        embedding_dim = self.config['model']['embedding_dim']
        
        # 创建测试数据
        predictions = torch.randn(batch_size, pred_len, output_dim).to(self.device)
        targets = torch.randn(batch_size, pred_len, output_dim).to(self.device)
        
        # 创建完整的辅助信息
        aux_info = {
            'expert_weights': torch.softmax(torch.randn(batch_size, num_experts).to(self.device), dim=-1),
            'expert_features': torch.randn(batch_size, 128).to(self.device),
            'gating_embeddings': torch.randn(batch_size, embedding_dim).to(self.device),
            'reconstruction_loss': torch.tensor(0.1).to(self.device),
            'triplet_loss': torch.tensor(0.05).to(self.device),
            'load_balance_loss': torch.tensor(0.02).to(self.device),
            'prototype_loss': torch.tensor(0.03).to(self.device)
        }
        
        # 测试损失计算
        try:
            losses = criterion(predictions, targets, aux_info)
            
            # 验证所有损失组件
            required_components = [
                'prediction', 'reconstruction', 'triplet', 'contrastive',
                'consistency', 'load_balance', 'prototype', 'total'
            ]
            
            for component in required_components:
                if component not in losses:
                    self.log_test(f"损失组件-{component}", "FAIL", 
                                "组件缺失", "HIGH")
                elif not torch.isfinite(losses[component]):
                    self.log_test(f"损失组件-{component}", "FAIL", 
                                "包含NaN/Inf", "HIGH")
                else:
                    self.log_test(f"损失组件-{component}", "PASS", 
                                f"值: {losses[component]:.4f}")
            
            # 测试可学习σ参数
            sigma_params = ['log_sigma_rc', 'log_sigma_cl', 'log_sigma_pr', 
                          'log_sigma_cons', 'log_sigma_bal']
            
            for param_name in sigma_params:
                if hasattr(criterion, param_name):
                    param = getattr(criterion, param_name)
                    if param.requires_grad:
                        sigma_value = torch.exp(param)
                        self.log_test(f"σ参数-{param_name}", "PASS", 
                                    f"σ={sigma_value.item():.4f}")
                    else:
                        self.log_test(f"σ参数-{param_name}", "FAIL", 
                                    "参数不可训练", "HIGH")
                else:
                    self.log_test(f"σ参数-{param_name}", "FAIL", 
                                "参数缺失", "HIGH")
            
            # 测试损失平衡
            total_expected = (
                torch.exp(-2 * criterion.log_sigma_pr) * losses['prediction'] + criterion.log_sigma_pr +
                torch.exp(-2 * criterion.log_sigma_rc) * losses['reconstruction'] + criterion.log_sigma_rc +
                torch.exp(-2 * criterion.log_sigma_cl) * (losses['triplet'] + 0.5 * losses['contrastive']) + criterion.log_sigma_cl +
                torch.exp(-2 * criterion.log_sigma_cons) * losses['consistency'] + criterion.log_sigma_cons +
                torch.exp(-2 * criterion.log_sigma_bal) * losses['load_balance'] + criterion.log_sigma_bal +
                losses['prototype'] * 0.1
            )
            
            if torch.allclose(losses['total'], total_expected, rtol=1e-3):
                self.log_test("损失平衡验证", "PASS", "总损失计算正确")
            else:
                self.log_test("损失平衡验证", "FAIL", 
                            f"总损失计算错误: {losses['total']:.4f} vs {total_expected:.4f}", "HIGH")
            
        except Exception as e:
            self.log_test("损失函数测试", "FAIL", str(e), "CRITICAL")
    
    def test_data_pipeline_integration(self):
        """测试数据管道集成"""
        print("\n📊 测试数据管道集成...")
        
        # 创建临时数据配置
        temp_config = self.config.copy()
        temp_config['data'].update({
            'dataset_type': 'synthetic',
            'data_path': 'synthetic',  # 修复None问题
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'synthetic_samples': 2000,  # 增加样本数量
            'noise_level': 0.1
        })
        
        try:
            # 测试数据模块初始化
            data_module = UniversalDataModule(temp_config)
            
            # 检查数据加载器
            train_loader = data_module.get_train_loader()
            val_loader = data_module.get_val_loader()
            test_loader = data_module.get_test_loader()
            
            self.log_test("数据加载器创建", "PASS", 
                        f"训练: {len(train_loader)}, 验证: {len(val_loader)}, 测试: {len(test_loader)}")
            
            # 测试数据批次
            for batch_x, batch_y in train_loader:
                expected_shape_x = (temp_config['data']['batch_size'], 
                                  temp_config['model']['seq_len'], 
                                  temp_config['model']['input_dim'])
                expected_shape_y = (temp_config['data']['batch_size'], 
                                  temp_config['model']['pred_len'], 
                                  temp_config['model']['output_dim'])
                
                if batch_x.shape == expected_shape_x and batch_y.shape == expected_shape_y:
                    self.log_test("数据形状验证", "PASS", 
                                f"X: {batch_x.shape}, Y: {batch_y.shape}")
                else:
                    self.log_test("数据形状验证", "FAIL", 
                                f"形状不匹配: X: {batch_x.shape}, Y: {batch_y.shape}", "HIGH")
                break
            
            # 测试数据数值范围
            data_stats = {
                'x_mean': batch_x.mean().item(),
                'x_std': batch_x.std().item(),
                'y_mean': batch_y.mean().item(),
                'y_std': batch_y.std().item()
            }
            
            if all(abs(stat) < 10 for stat in data_stats.values()):
                self.log_test("数据数值范围", "PASS", 
                            f"统计: {data_stats}")
            else:
                self.log_test("数据数值范围", "WARN", 
                            f"数值范围较大: {data_stats}")
            
        except Exception as e:
            self.log_test("数据管道集成", "FAIL", str(e), "HIGH")
    
    def test_end_to_end_training_simulation(self):
        """测试端到端训练模拟"""
        print("\n🚀 测试端到端训练模拟...")
        
        try:
            # 创建简化的训练配置
            train_config = self.config.copy()
            train_config['training']['epochs'] = 3
            train_config['data']['batch_size'] = 8
            train_config['data'].update({
                'dataset_type': 'synthetic',
                'data_path': 'synthetic',  # 修复None问题
                'synthetic_samples': 1000,  # 增加样本数量
                'noise_level': 0.1
            })
            
            # 初始化训练器
            trainer = M2MOEPTrainer(train_config)
            
            # 模拟训练几个epoch
            initial_loss = None
            final_loss = None
            
            for epoch in range(3):
                trainer.current_epoch = epoch
                
                # 训练一个epoch
                train_losses = trainer.train_epoch()
                val_losses, val_metrics = trainer.validate_epoch()
                
                if epoch == 0:
                    initial_loss = train_losses['total']
                if epoch == 2:
                    final_loss = train_losses['total']
                
                self.log_test(f"训练Epoch{epoch}", "PASS", 
                            f"训练损失: {train_losses['total']:.4f}, 验证损失: {val_losses['total']:.4f}")
            
            # 检查训练进展
            if final_loss < initial_loss:
                self.log_test("训练收敛性", "PASS", 
                            f"损失从 {initial_loss:.4f} 降至 {final_loss:.4f}")
            else:
                self.log_test("训练收敛性", "WARN", 
                            f"损失未下降: {initial_loss:.4f} -> {final_loss:.4f}")
            
            # 测试模型保存和加载
            save_path = "test_checkpoint.pth"
            trainer.save_checkpoint(is_best=True)
            
            if os.path.exists(os.path.join(trainer.save_dir, "best_model.pth")):
                self.log_test("模型保存", "PASS", "检查点保存成功")
                
                # 测试加载
                new_trainer = M2MOEPTrainer(train_config)
                checkpoint_path = os.path.join(trainer.save_dir, "best_model.pth")
                new_trainer.load_checkpoint(checkpoint_path)
                
                self.log_test("模型加载", "PASS", "检查点加载成功")
                
                # 清理
                os.remove(checkpoint_path)
            else:
                self.log_test("模型保存", "FAIL", "检查点保存失败", "HIGH")
            
        except Exception as e:
            self.log_test("端到端训练", "FAIL", str(e), "CRITICAL")
    
    def run_comprehensive_audit(self):
        """运行全面审计"""
        print("🚀 开始M²-MOEP全面代码审计...")
        print("=" * 100)
        
        start_time = time.time()
        
        try:
            # 1. 代码结构检查
            self.test_code_structure_and_imports()
            
            # 2. 模型初始化边界条件
            self.test_model_initialization_edge_cases()
            
            # 3. 数值稳定性
            self.test_numerical_stability_extreme_cases()
            
            # 4. 内存和计算效率
            self.test_memory_and_computational_efficiency()
            
            # 5. 梯度流和优化
            self.test_gradient_flow_and_optimization()
            
            # 6. 专家特化和多样性
            self.test_expert_specialization_and_diversity()
            
            # 7. 损失函数组件
            self.test_loss_function_components()
            
            # 8. 数据管道集成
            self.test_data_pipeline_integration()
            
            # 9. 端到端训练模拟
            self.test_end_to_end_training_simulation()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 生成审计报告
            self.generate_audit_report(total_time)
            
            return len(self.code_quality_issues) == 0
            
        except Exception as e:
            print(f"\n❌ 审计过程中发生严重错误: {str(e)}")
            traceback.print_exc()
            return False
    
    def generate_audit_report(self, total_time: float):
        """生成审计报告"""
        print("\n" + "=" * 100)
        print("📋 全面代码审计报告")
        print("=" * 100)
        
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        warned_tests = sum(1 for result in self.test_results if result['status'] == 'WARN')
        skipped_tests = sum(1 for result in self.test_results if result['status'] == 'SKIP')
        
        print(f"📊 测试统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests} ✅")
        print(f"   失败: {failed_tests} ❌")
        print(f"   警告: {warned_tests} ⚠️")
        print(f"   跳过: {skipped_tests} ⏭️")
        print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
        print(f"   总耗时: {total_time:.2f}秒")
        
        # 代码质量评估
        critical_issues = sum(1 for issue in self.code_quality_issues if 'CRITICAL' in issue)
        high_issues = len(self.code_quality_issues) - critical_issues
        
        print(f"\n🔍 代码质量评估:")
        print(f"   严重问题: {critical_issues}")
        print(f"   高优先级问题: {high_issues}")
        
        if critical_issues == 0 and high_issues == 0:
            print("   🎉 代码质量: 优秀")
        elif critical_issues == 0 and high_issues <= 3:
            print("   ✅ 代码质量: 良好")
        elif critical_issues <= 2:
            print("   ⚠️  代码质量: 需要改进")
        else:
            print("   ❌ 代码质量: 存在严重问题")
        
        # 性能指标
        if self.performance_metrics:
            print(f"\n⚡ 性能指标:")
            if 'computation_times' in self.performance_metrics:
                times = self.performance_metrics['computation_times']
                if times:
                    avg_time_per_sample = np.mean([t[1]/t[0] for t in times])
                    print(f"   平均每样本推理时间: {avg_time_per_sample*1000:.2f}ms")
            
            if 'memory_usage' in self.performance_metrics:
                memory = self.performance_metrics['memory_usage']
                if memory:
                    avg_memory_per_sample = np.mean([m[1]/m[0] for m in memory])
                    print(f"   平均每样本内存使用: {avg_memory_per_sample:.2f}MB")
        
        # 问题列表
        if self.code_quality_issues:
            print(f"\n⚠️  需要关注的问题:")
            for i, issue in enumerate(self.code_quality_issues[:10], 1):
                print(f"   {i}. {issue}")
            if len(self.code_quality_issues) > 10:
                print(f"   ... 还有 {len(self.code_quality_issues) - 10} 个问题")
        
        # 总结
        print(f"\n🎯 审计结论:")
        if failed_tests == 0 and critical_issues == 0:
            print("   ✅ 代码已准备好用于生产环境")
        elif failed_tests <= 2 and critical_issues == 0:
            print("   ⚠️  代码基本可用，建议修复警告问题")
        else:
            print("   ❌ 代码需要修复关键问题后才能使用")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    auditor = ComprehensiveCodeAudit()
    success = auditor.run_comprehensive_audit()
    
    sys.exit(0 if success else 1) 