"""
M²-MOEP 全方位测试套件
测试覆盖：Flow模型、门控、专家网络、损失函数、Top-k稀疏、可学习Δ等
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import warnings

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


class ComprehensiveTestSuite:
    """全方位测试套件"""
    
    def __init__(self):
        self.test_results = []
        self.config = self._get_test_config()
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
    def _get_test_config(self):
        """测试配置"""
        return {
            'model': {
                'input_dim': 6,
                'output_dim': 6,
                'hidden_dim': 128,
                'num_experts': 4,
                'seq_len': 24,
                'pred_len': 12,
                'expert_params': {
                    'mamba_d_model': 128,
                    'mamba_scales': [1, 2, 4]
                },
                'flow': {
                    'latent_dim': 64,
                    'use_pretrained': False,
                    'hidden_dim': 128,
                    'num_coupling_layers': 4
                },
                'diversity': {
                    'prototype_dim': 32,
                    'num_prototypes': 8,
                    'force_diversity': True
                },
                'temperature': {
                    'initial': 1.0,
                    'min': 0.1,
                    'max': 10.0
                },
                'triplet': {
                    'margin': 0.5,
                    'mining_strategy': 'batch_hard'
                },
                'top_k': 3,  # 启用Top-k稀疏
                'embedding_dim': 64
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
                'aux_loss_weight': 0.01
            },
            'data': {
                'seq_len': 24,
                'pred_len': 12,
                'batch_size': 8
            }
        }
    
    def log_test(self, test_name, status, details=""):
        """记录测试结果"""
        result = {
            'test': test_name,
            'status': status,
            'details': details
        }
        self.test_results.append(result)
        status_icon = "✅" if status == "PASS" else "❌"
        print(f"{status_icon} {test_name}: {details}")
    
    def test_flow_model(self):
        """测试Flow模型功能"""
        try:
            # 创建Flow模型
            input_dim = self.config['model']['seq_len'] * self.config['model']['input_dim']
            flow = PowerfulNormalizingFlow(
                input_dim=input_dim,
                latent_dim=self.config['model']['flow']['latent_dim'],
                hidden_dim=self.config['model']['flow']['hidden_dim']
            ).to(self.device)
            
            batch_size = 4
            x = torch.randn(batch_size, input_dim).to(self.device)
            
            # 测试编码
            z = flow.encode(x)
            assert z.shape == (batch_size, self.config['model']['flow']['latent_dim'])
            
            # 测试重构
            x_recon = flow.reconstruct(x)
            assert x_recon.shape == x.shape
            
            # 测试数值稳定性
            assert torch.isfinite(z).all(), "Flow编码包含NaN/Inf"
            assert torch.isfinite(x_recon).all(), "Flow重构包含NaN/Inf"
            
            # 测试FlowLayer的log_det修复
            flow_layer = FlowLayer(input_dim).to(self.device)
            z_layer, log_det = flow_layer(x)
            assert log_det.shape == (batch_size,), f"log_det形状错误: {log_det.shape}"
            
            self.log_test("Flow模型测试", "PASS", f"编码: {z.shape}, 重构: {x_recon.shape}")
            
        except Exception as e:
            self.log_test("Flow模型测试", "FAIL", str(e))
    
    def test_gating_network(self):
        """测试门控网络"""
        try:
            gating = GatingEncoder(self.config).to(self.device)
            
            batch_size = 4
            latent_dim = self.config['model']['flow']['latent_dim']
            z_latent = torch.randn(batch_size, latent_dim).to(self.device)
            
            # 测试前向传播
            gating_output = gating(z_latent)
            assert gating_output.shape == (batch_size, self.config['model']['num_experts'])
            
            # 测试嵌入提取
            embeddings = gating.get_embeddings(z_latent)
            assert embeddings.shape == (batch_size, self.config['model']['embedding_dim'])
            
            # 测试专家权重计算
            expert_weights = F.softmax(gating_output, dim=-1)
            assert torch.allclose(expert_weights.sum(dim=-1), torch.ones(batch_size).to(self.device), atol=1e-6)
            
            self.log_test("门控网络测试", "PASS", f"输出: {gating_output.shape}, 嵌入: {embeddings.shape}")
            
        except Exception as e:
            self.log_test("门控网络测试", "FAIL", str(e))
    
    def test_expert_network(self):
        """测试专家网络（包括可学习Δ）"""
        try:
            expert = FFTmsMambaExpert(self.config).to(self.device)
            
            batch_size = 4
            seq_len = self.config['model']['seq_len']
            input_dim = self.config['model']['input_dim']
            pred_len = self.config['model']['pred_len']
            
            x = torch.randn(batch_size, seq_len, input_dim).to(self.device)
            
            # 测试前向传播
            output = expert(x)
            expected_shape = (batch_size, pred_len, self.config['model']['output_dim'])
            assert output.shape == expected_shape, f"专家输出形状错误: {output.shape} vs {expected_shape}"
            
            # 测试可学习Δ参数
            assert hasattr(expert, 'learnable_deltas'), "缺少可学习Δ参数"
            assert expert.learnable_deltas.requires_grad, "Δ参数不可训练"
            
            # 测试FFT融合
            projected = expert.input_projection(x)
            fused_features = expert._early_fft_fusion(projected)
            expected_fused_shape = (batch_size, seq_len, expert.d_model)
            assert fused_features.shape == expected_fused_shape
            
            self.log_test("专家网络测试", "PASS", f"输出: {output.shape}, FFT融合: {fused_features.shape}")
            
        except Exception as e:
            self.log_test("专家网络测试", "FAIL", str(e))
    
    def test_top_k_sparse_gating(self):
        """测试Top-k稀疏门控"""
        try:
            model = M2_MOEP(self.config).to(self.device)
            model.eval()
            
            batch_size = 4
            seq_len = self.config['model']['seq_len']
            input_dim = self.config['model']['input_dim']
            
            x = torch.randn(batch_size, seq_len, input_dim).to(self.device)
            
            with torch.no_grad():
                output = model(x, return_aux_info=True)
                expert_weights = output['aux_info']['expert_weights']
                
                # 检查Top-k稀疏性
                top_k = self.config['model']['top_k']
                for i in range(batch_size):
                    non_zero_count = (expert_weights[i] > 1e-8).sum().item()
                    assert non_zero_count <= top_k, f"样本{i}激活了{non_zero_count}个专家，超过top_k={top_k}"
                
                # 检查权重归一化
                weight_sums = expert_weights.sum(dim=-1)
                assert torch.allclose(weight_sums, torch.ones(batch_size).to(self.device), atol=1e-6), "Top-k权重未正确归一化"
            
            self.log_test("Top-k稀疏门控测试", "PASS", f"每样本最多激活{top_k}个专家")
            
        except Exception as e:
            self.log_test("Top-k稀疏门控测试", "FAIL", str(e))
    
    def test_uncertainty_weighted_loss(self):
        """测试不确定性加权损失"""
        try:
            criterion = CompositeLoss(self.config).to(self.device)
            
            # 检查可学习σ参数
            sigma_params = ['log_sigma_rc', 'log_sigma_cl', 'log_sigma_pr', 'log_sigma_cons', 'log_sigma_bal']
            for param_name in sigma_params:
                assert hasattr(criterion, param_name), f"缺少{param_name}参数"
                param = getattr(criterion, param_name)
                assert param.requires_grad, f"{param_name}不可训练"
            
            # 测试损失计算
            batch_size = 4
            pred_len = self.config['model']['pred_len']
            output_dim = self.config['model']['output_dim']
            
            predictions = torch.randn(batch_size, pred_len, output_dim).to(self.device)
            targets = torch.randn(batch_size, pred_len, output_dim).to(self.device)
            
            # 模拟辅助信息
            aux_info = {
                'expert_weights': torch.softmax(torch.randn(batch_size, 4).to(self.device), dim=-1),
                'expert_features': torch.randn(batch_size, 32).to(self.device),
                'gating_embeddings': torch.randn(batch_size, 64).to(self.device),
                'reconstruction_loss': torch.tensor(0.1).to(self.device),
                'triplet_loss': torch.tensor(0.05).to(self.device),
                'load_balance_loss': torch.tensor(0.02).to(self.device),
                'prototype_loss': torch.tensor(0.03).to(self.device)
            }
            
            losses = criterion(predictions, targets, aux_info)
            
            # 检查所有损失项
            expected_keys = ['prediction', 'reconstruction', 'triplet', 'contrastive', 
                           'consistency', 'load_balance', 'prototype', 'total']
            for key in expected_keys:
                assert key in losses, f"缺少损失项: {key}"
                assert torch.isfinite(losses[key]), f"{key}损失包含NaN/Inf"
            
            self.log_test("不确定性加权损失测试", "PASS", f"总损失: {losses['total']:.4f}")
            
        except Exception as e:
            self.log_test("不确定性加权损失测试", "FAIL", str(e))
    
    def test_kl_consistency_loss(self):
        """测试KL一致性损失"""
        try:
            criterion = CompositeLoss(self.config).to(self.device)
            
            batch_size = 6
            num_experts = 4
            embedding_dim = 64
            
            # 创建相似的嵌入对
            embeddings = torch.randn(batch_size, embedding_dim).to(self.device)
            embeddings[1] = embeddings[0] + 0.1 * torch.randn(embedding_dim).to(self.device)  # 相似样本
            
            routing_weights = torch.softmax(torch.randn(batch_size, num_experts).to(self.device), dim=-1)
            routing_weights[1] = routing_weights[0] + 0.05 * torch.randn(num_experts).to(self.device)  # 相似路由
            routing_weights[1] = torch.softmax(routing_weights[1], dim=-1)
            
            # 计算KL一致性损失
            kl_loss = criterion.compute_kl_consistency_loss(routing_weights, embeddings)
            
            assert torch.isfinite(kl_loss), "KL一致性损失包含NaN/Inf"
            assert kl_loss >= 0, "KL散度应为非负"
            
            self.log_test("KL一致性损失测试", "PASS", f"KL损失: {kl_loss:.4f}")
            
        except Exception as e:
            self.log_test("KL一致性损失测试", "FAIL", str(e))
    
    def test_triplet_mining(self):
        """测试基于预测性能的三元组挖掘"""
        try:
            model = M2_MOEP(self.config).to(self.device)
            
            batch_size = 6
            seq_len = self.config['model']['seq_len']
            input_dim = self.config['model']['input_dim']
            pred_len = self.config['model']['pred_len']
            output_dim = self.config['model']['output_dim']
            num_experts = self.config['model']['num_experts']
            
            x = torch.randn(batch_size, seq_len, input_dim).to(self.device)
            ground_truth = torch.randn(batch_size, pred_len, output_dim).to(self.device)
            expert_weights = torch.softmax(torch.randn(batch_size, num_experts).to(self.device), dim=-1)
            expert_predictions = torch.randn(batch_size, num_experts, pred_len, output_dim).to(self.device)
            
            # 测试三元组挖掘
            triplets = model.mine_triplets_based_on_prediction_performance(
                x, expert_weights, expert_predictions, ground_truth
            )
            
            # 验证三元组格式
            for triplet in triplets:
                assert len(triplet) == 3, "三元组应包含3个元素"
                anchor, pos, neg = triplet
                assert 0 <= anchor < batch_size
                assert 0 <= pos < batch_size
                assert 0 <= neg < batch_size
                assert anchor != pos and anchor != neg, "锚点不应与正负样本相同"
            
            self.log_test("三元组挖掘测试", "PASS", f"挖掘到{len(triplets)}个三元组")
            
        except Exception as e:
            self.log_test("三元组挖掘测试", "FAIL", str(e))
    
    def test_end_to_end_training(self):
        """测试端到端训练流程"""
        try:
            model = M2_MOEP(self.config).to(self.device)
            criterion = CompositeLoss(self.config).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            batch_size = 4
            seq_len = self.config['model']['seq_len']
            input_dim = self.config['model']['input_dim']
            pred_len = self.config['model']['pred_len']
            output_dim = self.config['model']['output_dim']
            
            # 模拟训练步骤
            model.train()
            for step in range(3):
                optimizer.zero_grad()
                
                x = torch.randn(batch_size, seq_len, input_dim).to(self.device)
                y = torch.randn(batch_size, pred_len, output_dim).to(self.device)
                
                # 前向传播
                output = model(x, ground_truth=y, return_aux_info=True)
                predictions = output['predictions']
                aux_info = output['aux_info']
                
                # 损失计算
                losses = criterion(predictions, y, aux_info)
                
                # 反向传播
                losses['total'].backward()
                
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                assert torch.isfinite(grad_norm), f"步骤{step}梯度包含NaN/Inf"
                
                optimizer.step()
                
                # 温度调度
                expert_entropy = -torch.sum(
                    aux_info['expert_weights'].mean(0) * 
                    torch.log(aux_info['expert_weights'].mean(0) + 1e-8)
                )
                model.update_temperature_schedule(step, expert_entropy)
            
            self.log_test("端到端训练测试", "PASS", f"完成3步训练，最终损失: {losses['total']:.4f}")
            
        except Exception as e:
            self.log_test("端到端训练测试", "FAIL", str(e))
    
    def test_metrics_calculation(self):
        """测试指标计算"""
        try:
            batch_size = 8
            pred_len = 12
            input_dim = 6
            
            # 生成测试数据
            predictions = torch.randn(batch_size, pred_len, input_dim)
            targets = torch.randn(batch_size, pred_len, input_dim)
            expert_weights = torch.softmax(torch.randn(batch_size, 4), dim=-1)
            
            # 测试预测指标
            pred_metrics = calculate_metrics(predictions.numpy(), targets.numpy())
            expected_metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
            for metric in expected_metrics:
                assert metric in pred_metrics, f"缺少指标: {metric}"
                assert np.isfinite(pred_metrics[metric]), f"{metric}包含NaN/Inf"
            
            # 测试专家指标
            expert_metrics = compute_expert_metrics(expert_weights)
            expected_expert_metrics = ['expert_entropy', 'normalized_entropy', 'gini_coefficient', 'active_experts']
            for metric in expected_expert_metrics:
                assert metric in expert_metrics, f"缺少专家指标: {metric}"
                assert np.isfinite(expert_metrics[metric]), f"专家{metric}包含NaN/Inf"
            
            self.log_test("指标计算测试", "PASS", f"RMSE: {pred_metrics['RMSE']:.4f}, 专家熵: {expert_metrics['expert_entropy']:.4f}")
            
        except Exception as e:
            self.log_test("指标计算测试", "FAIL", str(e))
    
    def test_model_serialization(self):
        """测试模型序列化"""
        try:
            model = M2_MOEP(self.config)
            
            # 保存模型
            torch.save(model.state_dict(), '/tmp/test_model.pth')
            
            # 加载模型
            model_new = M2_MOEP(self.config)
            model_new.load_state_dict(torch.load('/tmp/test_model.pth'))
            
            # 验证参数一致性
            for (name1, param1), (name2, param2) in zip(model.named_parameters(), model_new.named_parameters()):
                assert name1 == name2, f"参数名不匹配: {name1} vs {name2}"
                assert torch.allclose(param1, param2), f"参数值不匹配: {name1}"
            
            # 清理
            os.remove('/tmp/test_model.pth')
            
            self.log_test("模型序列化测试", "PASS", "保存和加载成功")
            
        except Exception as e:
            self.log_test("模型序列化测试", "FAIL", str(e))
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始M²-MOEP全方位测试...")
        print("=" * 60)
        
        # 执行所有测试
        test_methods = [
            self.test_flow_model,
            self.test_gating_network,
            self.test_expert_network,
            self.test_top_k_sparse_gating,
            self.test_uncertainty_weighted_loss,
            self.test_kl_consistency_loss,
            self.test_triplet_mining,
            self.test_end_to_end_training,
            self.test_metrics_calculation,
            self.test_model_serialization
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log_test(test_method.__name__, "FAIL", f"未捕获异常: {str(e)}")
        
        # 统计结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        print("=" * 60)
        print(f"📊 测试结果汇总:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests} ✅")
        print(f"   失败: {failed_tests} ❌")
        print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests == 0:
            print("🎉 所有测试通过！项目代码质量良好。")
        else:
            print("⚠️  部分测试失败，请检查上述错误信息。")
            
        return failed_tests == 0


if __name__ == "__main__":
    # 忽略警告
    warnings.filterwarnings("ignore")
    
    test_suite = ComprehensiveTestSuite()
    success = test_suite.run_all_tests()
    
    # 退出码
    sys.exit(0 if success else 1) 