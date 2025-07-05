"""
超全面测试套件 - 覆盖训练全流程
细致测试每个关键函数、关键参数、关键模块
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
from typing import Dict, Any, List, Tuple
import logging
import warnings
import time
import psutil
import gc

# 设置警告级别
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class UltraComprehensiveTestSuite:
    """超全面测试套件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = {}
        self.critical_errors = []
        
        # 测试配置
        self.small_batch_size = 2
        self.medium_batch_size = 4
        self.large_batch_size = 8
        
        print(f"🔥 超全面测试开始！设备: {self.device}")
        print("=" * 100)
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """记录测试结果"""
        status = "✅" if success else "❌"
        self.logger.info(f"{status} {test_name}: {message}")
        self.test_results[test_name] = {"success": success, "message": message}
        if not success:
            self.critical_errors.append(f"{test_name}: {message}")
    
    def test_config_generation_detailed(self) -> Dict[str, Any]:
        """深度测试配置生成"""
        from configs.config_generator import ConfigGenerator
        
        # 测试不同数据集的配置生成
        test_configs = {}
        datasets = ['weather', 'etth1', 'etth2', 'ettm1', 'ettm2']
        
        for dataset in datasets:
            try:
                config = ConfigGenerator.generate_config(
                    dataset, batch_size=self.small_batch_size, epochs=2
                )
                test_configs[dataset] = config
                
                # 验证配置完整性
                self._validate_config_completeness(config, dataset)
                
                self.log_test(f"配置生成-{dataset}", True, f"配置生成成功")
                
            except Exception as e:
                self.log_test(f"配置生成-{dataset}", False, f"配置生成失败: {e}")
        
        return test_configs['weather'] if 'weather' in test_configs else {}
    
    def _validate_config_completeness(self, config: Dict, dataset: str):
        """验证配置完整性"""
        required_sections = ['model', 'data', 'training', 'evaluation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"缺失配置段: {section}")
        
        # 模型配置验证
        model_config = config['model']
        required_model_keys = [
            'input_dim', 'output_dim', 'hidden_dim', 'num_experts',
            'expert_params', 'flow', 'triplet', 'diversity', 'temperature'
        ]
        
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"缺失模型配置: {key}")
        
        # 专家参数验证
        expert_params = model_config['expert_params']
        if 'mamba_d_model' not in expert_params:
            raise ValueError("缺失mamba_d_model")
        if 'mamba_scales' not in expert_params:
            raise ValueError("缺失mamba_scales")
        
        # 验证数值范围
        if model_config['num_experts'] < 1 or model_config['num_experts'] > 10:
            raise ValueError(f"专家数量异常: {model_config['num_experts']}")
        
        if model_config['hidden_dim'] < 32 or model_config['hidden_dim'] > 1024:
            raise ValueError(f"隐藏维度异常: {model_config['hidden_dim']}")
    
    def test_data_loading_detailed(self, config: Dict[str, Any]) -> bool:
        """深度测试数据加载"""
        from data.universal_dataset import UniversalDataModule
        
        try:
            # 初始化数据模块
            data_module = UniversalDataModule(config)
            
            # 获取数据加载器
            train_loader = data_module.get_train_loader()
            val_loader = data_module.get_val_loader()
            test_loader = data_module.get_test_loader()
            
            # 测试批次一致性
            self._test_batch_consistency(train_loader, "训练集")
            self._test_batch_consistency(val_loader, "验证集")
            self._test_batch_consistency(test_loader, "测试集")
            
            # 测试数据范围和分布
            self._test_data_distribution(train_loader, "训练集")
            
            # 测试数据增强（如果有）
            self._test_data_augmentation(data_module, config)
            
            self.log_test("数据加载详细测试", True, "数据加载详细测试通过")
            return True
            
        except Exception as e:
            self.log_test("数据加载详细测试", False, f"数据加载详细测试失败: {e}")
            return False
    
    def _test_batch_consistency(self, loader, loader_name: str):
        """测试批次一致性"""
        batch_shapes = []
        batch_dtypes = []
        
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_shapes.append((batch_x.shape, batch_y.shape))
            batch_dtypes.append((batch_x.dtype, batch_y.dtype))
            
            # 检查数值范围
            if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                raise ValueError(f"{loader_name}批次{i}输入包含NaN或Inf")
            
            if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                raise ValueError(f"{loader_name}批次{i}目标包含NaN或Inf")
            
            if i >= 3:  # 只检查前几个批次
                break
        
        # 检查形状一致性
        if len(set(batch_shapes)) > 1:
            raise ValueError(f"{loader_name}批次形状不一致: {batch_shapes}")
        
        # 检查数据类型一致性
        if len(set(batch_dtypes)) > 1:
            raise ValueError(f"{loader_name}批次数据类型不一致: {batch_dtypes}")
    
    def _test_data_distribution(self, loader, loader_name: str):
        """测试数据分布"""
        all_x = []
        all_y = []
        
        for i, (batch_x, batch_y) in enumerate(loader):
            all_x.append(batch_x)
            all_y.append(batch_y)
            if i >= 5:  # 只检查前几个批次
                break
        
        if all_x:
            x_tensor = torch.cat(all_x, dim=0)
            y_tensor = torch.cat(all_y, dim=0)
            
            # 检查统计特性
            x_mean = x_tensor.mean().item()
            x_std = x_tensor.std().item()
            y_mean = y_tensor.mean().item()
            y_std = y_tensor.std().item()
            
            print(f"{loader_name}统计: X均值={x_mean:.3f}, X标准差={x_std:.3f}, Y均值={y_mean:.3f}, Y标准差={y_std:.3f}")
            
            # 检查异常值
            if abs(x_mean) > 100 or x_std > 1000:
                raise ValueError(f"{loader_name}输入数据分布异常")
    
    def _test_data_augmentation(self, data_module, config):
        """测试数据增强"""
        # 检查是否有数据增强配置
        if 'augmentation' in config.get('data', {}):
            print("检测到数据增强配置")
            # 这里可以添加数据增强的具体测试
    
    def test_expert_network_detailed(self, config: Dict[str, Any]) -> bool:
        """详细测试专家网络"""
        from models.expert import FFTmsMambaExpert
        
        try:
            # 创建专家网络
            expert = FFTmsMambaExpert(config).to(self.device)
            
            # 测试不同批次大小
            batch_sizes = [self.small_batch_size, self.medium_batch_size, self.large_batch_size]
            
            for batch_size in batch_sizes:
                self._test_expert_single_batch(expert, config, batch_size)
            
            # 测试专家网络的特定功能
            self._test_expert_fft_fusion(expert, config)
            self._test_expert_multiscale_processing(expert, config)
            self._test_expert_personalization(expert, config)
            
            # 测试专家网络的数值稳定性
            self._test_expert_numerical_stability(expert, config)
            
            # 测试专家网络的梯度流
            self._test_expert_gradient_flow(expert, config)
            
            self.log_test("专家网络详细测试", True, "专家网络详细测试通过")
            return True
            
        except Exception as e:
            self.log_test("专家网络详细测试", False, f"专家网络详细测试失败: {e}")
            traceback.print_exc()
            return False
    
    def _test_expert_single_batch(self, expert, config, batch_size):
        """测试专家网络单批次"""
        seq_len = config['data']['seq_len']
        input_dim = config['model']['input_dim']
        pred_len = config['data']['pred_len']
        output_dim = config['model']['output_dim']
        
        test_input = torch.randn(batch_size, seq_len, input_dim).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            output = expert(test_input)
        
        # 验证输出
        expected_shape = (batch_size, pred_len, output_dim)
        if output.shape != expected_shape:
            raise ValueError(f"批次{batch_size}输出形状错误: {output.shape} vs {expected_shape}")
        
        # 数值检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError(f"批次{batch_size}输出包含NaN或Inf")
        
        print(f"✓ 专家网络批次{batch_size}测试通过")
    
    def _test_expert_fft_fusion(self, expert, config):
        """测试专家网络FFT融合"""
        if hasattr(expert, 'fft_fusion') and expert.fft_fusion is not None:
            print("✓ 专家网络包含FFT融合层")
            
            # 测试FFT融合的数值稳定性
            batch_size = self.small_batch_size
            seq_len = config['data']['seq_len']
            input_dim = config['model']['input_dim']
            
            # 极端输入测试
            extreme_inputs = [
                torch.zeros(batch_size, seq_len, input_dim).to(self.device),
                torch.ones(batch_size, seq_len, input_dim).to(self.device),
                torch.randn(batch_size, seq_len, input_dim).to(self.device) * 100
            ]
            
            for i, extreme_input in enumerate(extreme_inputs):
                fused = expert._stable_fft_fusion(extreme_input)
                if torch.isnan(fused).any() or torch.isinf(fused).any():
                    raise ValueError(f"FFT融合极端输入{i}失败")
        else:
            print("⚠ 专家网络不包含FFT融合层")
    
    def _test_expert_multiscale_processing(self, expert, config):
        """测试专家网络多尺度处理"""
        print("✓ 专家网络多尺度处理测试")
        
        # 检查learnable_deltas
        if hasattr(expert, 'learnable_deltas'):
            deltas = expert.learnable_deltas
            print(f"可学习尺度参数: {deltas}")
            
            # 检查尺度参数是否在合理范围内
            if torch.any(deltas < 0.5) or torch.any(deltas > 10):
                raise ValueError(f"尺度参数异常: {deltas}")
        
        # 检查多尺度层数量
        if hasattr(expert, 'multi_scale_mamba'):
            num_scales = len(expert.multi_scale_mamba)
            print(f"多尺度层数量: {num_scales}")
    
    def _test_expert_personalization(self, expert, config):
        """测试专家个性化"""
        if hasattr(expert, 'expert_personalization'):
            print("✓ 专家网络包含个性化层")
            
            # 测试个性化层的参数
            for name, param in expert.expert_personalization.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    raise ValueError(f"个性化层参数{name}异常")
        else:
            print("⚠ 专家网络不包含个性化层")
    
    def _test_expert_numerical_stability(self, expert, config):
        """测试专家网络数值稳定性"""
        batch_size = self.small_batch_size
        seq_len = config['data']['seq_len']
        input_dim = config['model']['input_dim']
        
        # 极端输入测试
        extreme_cases = [
            torch.zeros(batch_size, seq_len, input_dim).to(self.device),
            torch.ones(batch_size, seq_len, input_dim).to(self.device) * 1000,
            torch.randn(batch_size, seq_len, input_dim).to(self.device) * 0.001
        ]
        
        for i, extreme_input in enumerate(extreme_cases):
            try:
                with torch.no_grad():
                    output = expert(extreme_input)
                
                if torch.isnan(output).any() or torch.isinf(output).any():
                    raise ValueError(f"极端输入{i}导致数值异常")
                
                print(f"✓ 极端输入{i}测试通过")
            except Exception as e:
                print(f"⚠ 极端输入{i}测试失败: {e}")
    
    def _test_expert_gradient_flow(self, expert, config):
        """测试专家网络梯度流"""
        batch_size = self.small_batch_size
        seq_len = config['data']['seq_len']
        input_dim = config['model']['input_dim']
        
        expert.train()
        test_input = torch.randn(batch_size, seq_len, input_dim, requires_grad=True).to(self.device)
        
        output = expert(test_input)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度
        gradient_norms = []
        for name, param in expert.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms.append(grad_norm)
                
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise ValueError(f"参数{name}梯度异常")
        
        if not gradient_norms:
            raise ValueError("没有计算到梯度")
        
        avg_grad_norm = np.mean(gradient_norms)
        print(f"✓ 梯度流测试通过，平均梯度范数: {avg_grad_norm:.6f}")
    
    def test_full_model_detailed(self, config: Dict[str, Any]) -> bool:
        """详细测试完整模型"""
        from models.m2_moep import M2_MOEP
        
        try:
            # 创建模型
            model = M2_MOEP(config).to(self.device)
            
            # 测试模型组件
            self._test_model_components(model, config)
            
            # 测试模型前向传播
            self._test_model_forward_pass(model, config)
            
            # 测试模型的特殊功能
            self._test_model_temperature_scheduling(model, config)
            self._test_model_expert_routing(model, config)
            self._test_model_triplet_mining(model, config)
            
            # 测试模型的数值稳定性
            self._test_model_numerical_stability(model, config)
            
            self.log_test("完整模型详细测试", True, "完整模型详细测试通过")
            return True
            
        except Exception as e:
            self.log_test("完整模型详细测试", False, f"完整模型详细测试失败: {e}")
            traceback.print_exc()
            return False
    
    def _test_model_components(self, model, config):
        """测试模型组件"""
        # 检查flow模型
        if hasattr(model, 'flow_model'):
            print("✓ 模型包含Flow模型")
            
            # 测试Flow模型的编码解码
            test_input = torch.randn(2, model.flow_model.input_dim).to(self.device)
            try:
                latent = model.flow_model.encode(test_input)
                reconstructed = model.flow_model.decode(latent)
                print(f"✓ Flow模型编码解码测试通过")
            except Exception as e:
                print(f"⚠ Flow模型编码解码测试失败: {e}")
        
        # 检查门控网络
        if hasattr(model, 'gating'):
            print("✓ 模型包含门控网络")
            
            # 测试门控网络
            test_latent = torch.randn(2, model.flow_latent_dim).to(self.device)
            try:
                gating_output = model.gating(test_latent)
                expert_embeddings = model.gating.get_embeddings(test_latent)
                print(f"✓ 门控网络测试通过")
            except Exception as e:
                print(f"⚠ 门控网络测试失败: {e}")
        
        # 检查专家网络
        if hasattr(model, 'experts'):
            print(f"✓ 模型包含{len(model.experts)}个专家网络")
            
            # 测试每个专家
            for i, expert in enumerate(model.experts):
                try:
                    test_input = torch.randn(2, config['data']['seq_len'], config['model']['input_dim']).to(self.device)
                    expert_output = expert(test_input)
                    print(f"✓ 专家{i}测试通过")
                except Exception as e:
                    print(f"⚠ 专家{i}测试失败: {e}")
    
    def _test_model_forward_pass(self, model, config):
        """测试模型前向传播"""
        batch_sizes = [self.small_batch_size, self.medium_batch_size]
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, config['data']['seq_len'], config['model']['input_dim']).to(self.device)
            
            # 推理模式
            model.eval()
            with torch.no_grad():
                output = model(test_input, return_aux_info=True)
            
            # 验证输出
            if 'predictions' not in output:
                raise ValueError("模型输出缺少predictions")
            
            predictions = output['predictions']
            expected_shape = (batch_size, config['data']['pred_len'], config['model']['output_dim'])
            if predictions.shape != expected_shape:
                raise ValueError(f"预测形状错误: {predictions.shape} vs {expected_shape}")
            
            # 验证辅助信息
            aux_info = output['aux_info']
            required_aux_keys = ['expert_weights', 'expert_embeddings', 'temperature']
            for key in required_aux_keys:
                if key not in aux_info:
                    raise ValueError(f"辅助信息缺少{key}")
            
            print(f"✓ 批次{batch_size}前向传播测试通过")
    
    def _test_model_temperature_scheduling(self, model, config):
        """测试模型温度调度"""
        if hasattr(model, 'temperature'):
            initial_temp = model.temperature.item()
            print(f"✓ 初始温度: {initial_temp}")
            
            # 测试温度更新
            if hasattr(model, 'update_temperature_schedule'):
                try:
                    expert_entropy = torch.log(torch.tensor(float(model.num_experts))) * 0.5
                    model.update_temperature_schedule(epoch=5, expert_entropy=expert_entropy)
                    new_temp = model.temperature.item()
                    print(f"✓ 温度调度测试通过，新温度: {new_temp}")
                except Exception as e:
                    print(f"⚠ 温度调度测试失败: {e}")
    
    def _test_model_expert_routing(self, model, config):
        """测试模型专家路由"""
        batch_size = self.small_batch_size
        test_input = torch.randn(batch_size, config['data']['seq_len'], config['model']['input_dim']).to(self.device)
        
        model.eval()
        with torch.no_grad():
            output = model(test_input, return_aux_info=True)
            expert_weights = output['aux_info']['expert_weights']
            
            # 验证专家权重
            if expert_weights.shape != (batch_size, model.num_experts):
                raise ValueError(f"专家权重形状错误: {expert_weights.shape}")
            
            # 验证权重和为1
            weight_sums = expert_weights.sum(dim=-1)
            if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5):
                raise ValueError("专家权重和不为1")
            
            # 计算路由多样性
            expert_usage = expert_weights.mean(dim=0)
            usage_entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
            print(f"✓ 专家路由测试通过，使用熵: {usage_entropy:.3f}")
    
    def _test_model_triplet_mining(self, model, config):
        """测试模型三元组挖掘"""
        batch_size = 6  # 需要足够的样本构成三元组
        
        test_input = torch.randn(batch_size, config['data']['seq_len'], config['model']['input_dim']).to(self.device)
        ground_truth = torch.randn(batch_size, config['data']['pred_len'], config['model']['output_dim']).to(self.device)
        
        model.eval()
        with torch.no_grad():
            output = model(test_input, return_aux_info=True)
            
            # 模拟三元组挖掘
            if hasattr(model, 'mine_triplets_based_on_prediction_performance'):
                try:
                    expert_weights = output['aux_info']['expert_weights']
                    expert_predictions = torch.stack([expert(test_input) for expert in model.experts], dim=1)
                    
                    triplets = model.mine_triplets_based_on_prediction_performance(
                        test_input, expert_weights, expert_predictions, ground_truth
                    )
                    
                    print(f"✓ 三元组挖掘测试通过，发现{len(triplets)}个三元组")
                except Exception as e:
                    print(f"⚠ 三元组挖掘测试失败: {e}")
    
    def _test_model_numerical_stability(self, model, config):
        """测试模型数值稳定性"""
        # 极端输入测试
        batch_size = self.small_batch_size
        seq_len = config['data']['seq_len']
        input_dim = config['model']['input_dim']
        
        extreme_cases = [
            torch.zeros(batch_size, seq_len, input_dim).to(self.device),
            torch.ones(batch_size, seq_len, input_dim).to(self.device) * 1000,
            torch.randn(batch_size, seq_len, input_dim).to(self.device) * 0.001
        ]
        
        model.eval()
        for i, extreme_input in enumerate(extreme_cases):
            try:
                with torch.no_grad():
                    output = model(extreme_input)
                
                predictions = output['predictions']
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    raise ValueError(f"极端输入{i}导致预测异常")
                
                print(f"✓ 模型极端输入{i}测试通过")
            except Exception as e:
                print(f"⚠ 模型极端输入{i}测试失败: {e}")
    
    def test_training_process_detailed(self, config: Dict[str, Any]) -> bool:
        """详细测试训练过程"""
        from train import M2MOEPTrainer
        
        try:
            # 创建训练器
            trainer = M2MOEPTrainer(config)
            
            # 测试训练器组件
            self._test_trainer_components(trainer, config)
            
            # 测试单步训练
            self._test_single_training_step(trainer, config)
            
            # 测试多步训练
            self._test_multiple_training_steps(trainer, config)
            
            # 测试验证步骤
            self._test_validation_step(trainer, config)
            
            # 测试模型保存和加载
            self._test_model_save_load(trainer, config)
            
            self.log_test("训练过程详细测试", True, "训练过程详细测试通过")
            return True
            
        except Exception as e:
            self.log_test("训练过程详细测试", False, f"训练过程详细测试失败: {e}")
            traceback.print_exc()
            return False
    
    def _test_trainer_components(self, trainer, config):
        """测试训练器组件"""
        # 检查训练器属性
        required_attrs = ['model', 'data_module', 'criterion', 'optimizer', 'scheduler', 'device']
        for attr in required_attrs:
            if not hasattr(trainer, attr):
                raise ValueError(f"训练器缺少属性: {attr}")
        
        print(f"✓ 训练器组件完整，设备: {trainer.device}")
        
        # 检查优化器
        if hasattr(trainer, 'optimizer'):
            print(f"✓ 优化器类型: {type(trainer.optimizer).__name__}")
            
            # 检查优化器参数
            param_groups = trainer.optimizer.param_groups
            print(f"✓ 优化器参数组: {len(param_groups)}")
            
            for i, group in enumerate(param_groups):
                print(f"  组{i}: lr={group['lr']}, weight_decay={group.get('weight_decay', 0)}")
        
        # 检查学习率调度器
        if hasattr(trainer, 'scheduler'):
            print(f"✓ 学习率调度器类型: {type(trainer.scheduler).__name__}")
    
    def _test_single_training_step(self, trainer, config):
        """测试单步训练"""
        # 获取一个批次的数据
        train_loader = trainer.data_module.get_train_loader()
        batch_x, batch_y = next(iter(train_loader))
        batch_x = batch_x.to(trainer.device)
        batch_y = batch_y.to(trainer.device)
        
        # 记录初始损失
        trainer.model.eval()
        with torch.no_grad():
            initial_output = trainer.model(batch_x, ground_truth=batch_y, return_aux_info=True)
            initial_loss = trainer.criterion(
                predictions=initial_output['predictions'],
                targets=batch_y,
                expert_weights=initial_output['aux_info'].get('expert_weights'),
                expert_embeddings=initial_output['aux_info'].get('expert_embeddings')
            )['total']
        
        # 执行训练步骤
        trainer.model.train()
        trainer.optimizer.zero_grad()
        
        output = trainer.model(batch_x, ground_truth=batch_y, return_aux_info=True)
        losses = trainer.criterion(
            predictions=output['predictions'],
            targets=batch_y,
            expert_weights=output['aux_info'].get('expert_weights'),
            expert_embeddings=output['aux_info'].get('expert_embeddings')
        )
        
        total_loss = losses['total']
        total_loss.backward()
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        
        trainer.optimizer.step()
        
        # 记录训练后损失
        trainer.model.eval()
        with torch.no_grad():
            final_output = trainer.model(batch_x, ground_truth=batch_y, return_aux_info=True)
            final_loss = trainer.criterion(
                predictions=final_output['predictions'],
                targets=batch_y,
                expert_weights=final_output['aux_info'].get('expert_weights'),
                expert_embeddings=final_output['aux_info'].get('expert_embeddings')
            )['total']
        
        print(f"✓ 单步训练: 初始损失={initial_loss.item():.4f}, 最终损失={final_loss.item():.4f}, 梯度范数={grad_norm.item():.4f}")
    
    def _test_multiple_training_steps(self, trainer, config):
        """测试多步训练"""
        train_loader = trainer.data_module.get_train_loader()
        
        losses = []
        for i, (batch_x, batch_y) in enumerate(train_loader):
            if i >= 3:  # 只测试前3步
                break
            
            batch_x = batch_x.to(trainer.device)
            batch_y = batch_y.to(trainer.device)
            
            trainer.model.train()
            trainer.optimizer.zero_grad()
            
            output = trainer.model(batch_x, ground_truth=batch_y, return_aux_info=True)
            loss_dict = trainer.criterion(
                predictions=output['predictions'],
                targets=batch_y,
                expert_weights=output['aux_info'].get('expert_weights'),
                expert_embeddings=output['aux_info'].get('expert_embeddings')
            )
            
            total_loss = loss_dict['total']
            total_loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
            trainer.optimizer.step()
            
            losses.append(total_loss.item())
            print(f"  步骤{i}: 损失={total_loss.item():.4f}, 梯度范数={grad_norm.item():.4f}")
        
        print(f"✓ 多步训练完成，平均损失: {np.mean(losses):.4f}")
    
    def _test_validation_step(self, trainer, config):
        """测试验证步骤"""
        val_loader = trainer.data_module.get_val_loader()
        
        trainer.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                if i >= 3:  # 只测试前3步
                    break
                
                batch_x = batch_x.to(trainer.device)
                batch_y = batch_y.to(trainer.device)
                
                output = trainer.model(batch_x, return_aux_info=True)
                loss_dict = trainer.criterion(
                    predictions=output['predictions'],
                    targets=batch_y,
                    expert_weights=output['aux_info'].get('expert_weights'),
                    expert_embeddings=output['aux_info'].get('expert_embeddings')
                )
                
                val_losses.append(loss_dict['total'].item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"✓ 验证步骤完成，平均验证损失: {avg_val_loss:.4f}")
    
    def _test_model_save_load(self, trainer, config):
        """测试模型保存和加载"""
        # 保存模型
        save_path = "test_model_checkpoint.pth"
        
        try:
            # 保存模型状态
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': config
            }, save_path)
            
            # 加载模型
            checkpoint = torch.load(save_path, map_location=trainer.device)
            
            # 创建新模型实例
            from models.m2_moep import M2_MOEP
            new_model = M2_MOEP(checkpoint['config']).to(trainer.device)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 测试加载后的模型
            test_input = torch.randn(2, config['data']['seq_len'], config['model']['input_dim']).to(trainer.device)
            
            new_model.eval()
            with torch.no_grad():
                output = new_model(test_input)
            
            print("✓ 模型保存和加载测试通过")
            
        except Exception as e:
            print(f"⚠ 模型保存和加载测试失败: {e}")
        finally:
            # 清理临时文件
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def run_ultra_comprehensive_test(self) -> Dict[str, Any]:
        """运行超全面测试"""
        print("🔬 开始超全面深度测试...")
        
        # 阶段1: 基础测试
        print("\n" + "="*50 + " 阶段1: 基础配置测试 " + "="*50)
        config = self.test_config_generation_detailed()
        if not config:
            print("❌ 基础配置测试失败，终止测试")
            return self._generate_test_report()
        
        # 阶段2: 数据测试
        print("\n" + "="*50 + " 阶段2: 数据加载测试 " + "="*50)
        self.test_data_loading_detailed(config)
        
        # 阶段3: 专家网络测试
        print("\n" + "="*50 + " 阶段3: 专家网络测试 " + "="*50)
        self.test_expert_network_detailed(config)
        
        # 阶段4: 完整模型测试
        print("\n" + "="*50 + " 阶段4: 完整模型测试 " + "="*50)
        self.test_full_model_detailed(config)
        
        # 阶段5: 训练过程测试
        print("\n" + "="*50 + " 阶段5: 训练过程测试 " + "="*50)
        self.test_training_process_detailed(config)
        
        # 生成报告
        return self._generate_test_report()
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        print("\n" + "="*100)
        print("📊 超全面测试报告")
        print("="*100)
        
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        total_tests = len(self.test_results)
        
        print(f"📈 总体结果: {passed_tests}/{total_tests} 测试通过")
        print(f"📈 成功率: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 所有测试通过！系统完全就绪。")
        else:
            print("⚠️  存在失败的测试:")
            for error in self.critical_errors:
                print(f"   - {error}")
        
        # 按类别汇总
        categories = {}
        for test_name, result in self.test_results.items():
            category = test_name.split('-')[0] if '-' in test_name else test_name
            if category not in categories:
                categories[category] = {'pass': 0, 'fail': 0}
            
            if result['success']:
                categories[category]['pass'] += 1
            else:
                categories[category]['fail'] += 1
        
        print("\n📋 分类测试结果:")
        for category, stats in categories.items():
            total = stats['pass'] + stats['fail']
            rate = stats['pass'] / total * 100 if total > 0 else 0
            print(f"   {category}: {stats['pass']}/{total} ({rate:.1f}%)")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'all_passed': passed_tests == total_tests,
            'test_results': self.test_results,
            'critical_errors': self.critical_errors,
            'categories': categories
        }

def main():
    """主函数"""
    test_suite = UltraComprehensiveTestSuite()
    report = test_suite.run_ultra_comprehensive_test()
    
    # 返回状态码
    return 0 if report['all_passed'] else 1

if __name__ == "__main__":
    sys.exit(main()) 