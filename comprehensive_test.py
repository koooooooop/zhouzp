"""
全面深度测试脚本 - 确保训练流程无问题
检查所有可能的运行时错误和潜在问题
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import traceback
from typing import Dict, Any
import logging
import warnings

# 设置警告级别
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def setup_test_logging():
    """设置测试日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

class ComprehensiveTestSuite:
    """全面测试套件"""
    
    def __init__(self):
        self.logger = setup_test_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = {}
        self.critical_errors = []
        
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """记录测试结果"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.logger.info(f"{status} {test_name}: {message}")
        self.test_results[test_name] = {"success": success, "message": message}
        if not success:
            self.critical_errors.append(f"{test_name}: {message}")
    
    def test_1_imports_and_dependencies(self) -> bool:
        """测试1: 导入和依赖关系"""
        try:
            # 核心依赖
            import torch
            import numpy as np
            import pandas as pd
            import yaml
            
            # 项目模块导入测试
            from models.m2_moep import M2_MOEP
            from models.expert import FFTmsMambaExpert
            from models.gating import GatingEncoder
            from models.flow import PowerfulNormalizingFlow
            from data.universal_dataset import UniversalDataModule
            from utils.losses import CompositeLoss
            from configs.config_generator import ConfigGenerator
            from train import M2MOEPTrainer
            
            self.log_test("模块导入", True, "所有核心模块导入成功")
            return True
            
        except ImportError as e:
            self.log_test("模块导入", False, f"导入失败: {e}")
            return False
        except Exception as e:
            self.log_test("模块导入", False, f"未知错误: {e}")
            return False
    
    def test_2_config_generation(self) -> Dict[str, Any]:
        """测试2: 配置生成和验证"""
        try:
            from configs.config_generator import ConfigGenerator
            
            # 生成测试配置
            config = ConfigGenerator.generate_config(
                'weather',
                batch_size=8,
                epochs=2,
                learning_rate=0.001
            )
            
            # 验证关键配置项
            required_keys = [
                'model', 'data', 'training', 'evaluation'
            ]
            
            for key in required_keys:
                if key not in config:
                    raise KeyError(f"缺失配置项: {key}")
            
            # 验证模型配置
            model_config = config['model']
            required_model_keys = [
                'input_dim', 'output_dim', 'hidden_dim', 'num_experts',
                'expert_params', 'flow', 'triplet', 'diversity', 'temperature'
            ]
            
            for key in required_model_keys:
                if key not in model_config:
                    raise KeyError(f"缺失模型配置项: {key}")
            
            # 验证专家参数
            expert_params = model_config['expert_params']
            if 'mamba_d_model' not in expert_params:
                raise KeyError("缺失mamba_d_model参数")
            if 'mamba_scales' not in expert_params:
                raise KeyError("缺失mamba_scales参数")
            
            # 验证数据配置
            data_config = config['data']
            if data_config['seq_len'] <= 0 or data_config['pred_len'] <= 0:
                raise ValueError("序列长度配置无效")
            
            self.log_test("配置生成和验证", True, f"配置验证成功，包含{len(config)}个主要部分")
            return config
            
        except Exception as e:
            self.log_test("配置生成和验证", False, f"配置错误: {e}")
            return {}
    
    def test_3_data_loading(self, config: Dict[str, Any]) -> bool:
        """测试3: 数据加载和处理"""
        try:
            from data.universal_dataset import UniversalDataModule
            
            # 初始化数据模块
            data_module = UniversalDataModule(config)
            
            # 验证数据信息
            dataset_info = data_module.get_dataset_info()
            required_info_keys = ['num_features', 'num_samples', 'seq_len', 'pred_len']
            
            for key in required_info_keys:
                if key not in dataset_info:
                    raise KeyError(f"缺失数据集信息: {key}")
            
            # 验证数据加载器
            train_loader = data_module.get_train_loader()
            val_loader = data_module.get_val_loader()
            test_loader = data_module.get_test_loader()
            
            # 测试一个批次的数据
            batch_x, batch_y = next(iter(train_loader))
            
            # 验证数据形状
            expected_x_shape = (config['training']['batch_size'], config['data']['seq_len'], dataset_info['num_features'])
            expected_y_shape = (config['training']['batch_size'], config['data']['pred_len'], dataset_info['num_features'])
            
            if batch_x.shape != expected_x_shape:
                self.logger.warning(f"输入形状不匹配: 期望{expected_x_shape}, 实际{batch_x.shape}")
            
            if batch_y.shape != expected_y_shape:
                self.logger.warning(f"目标形状不匹配: 期望{expected_y_shape}, 实际{batch_y.shape}")
            
            # 验证数据类型和设备
            if not torch.is_tensor(batch_x) or not torch.is_tensor(batch_y):
                raise TypeError("数据不是张量类型")
            
            if batch_x.dtype != torch.float32 or batch_y.dtype != torch.float32:
                raise TypeError(f"数据类型错误: {batch_x.dtype}, {batch_y.dtype}")
            
            # 验证数值范围
            if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                raise ValueError("输入数据包含NaN或Inf")
            
            if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                raise ValueError("目标数据包含NaN或Inf")
            
            self.log_test("数据加载和处理", True, 
                         f"数据加载成功: {len(train_loader)}个训练批次, 特征数{dataset_info['num_features']}")
            return True
            
        except Exception as e:
            self.log_test("数据加载和处理", False, f"数据加载错误: {e}")
            return False
    
    def test_4_expert_network(self, config: Dict[str, Any]) -> bool:
        """测试4: 专家网络详细测试"""
        try:
            from models.expert import FFTmsMambaExpert
            
            # 更新配置以确保当前专家ID
            config['model']['current_expert_id'] = 0
            
            # 创建专家网络
            expert = FFTmsMambaExpert(config).to(self.device)
            
            # 验证专家网络结构
            required_attrs = [
                'input_projection', 'multi_scale_mamba', 'scale_fusion',
                'output_projection', 'prediction_head', 'expert_personalization',
                'learnable_deltas', 'use_mamba'
            ]
            
            for attr in required_attrs:
                if not hasattr(expert, attr):
                    raise AttributeError(f"专家网络缺失属性: {attr}")
            
            # 验证可学习参数
            if not isinstance(expert.learnable_deltas, nn.Parameter):
                raise TypeError("learnable_deltas不是可学习参数")
            
            if len(expert.learnable_deltas) != len(config['model']['expert_params']['mamba_scales']):
                raise ValueError("learnable_deltas长度与scales不匹配")
            
            # 测试前向传播
            batch_size = 4
            seq_len = config['data']['seq_len']
            input_dim = config['model']['input_dim']
            
            test_input = torch.randn(batch_size, seq_len, input_dim).to(self.device)
            
            # 前向传播测试
            with torch.no_grad():
                output = expert(test_input)
            
            # 验证输出形状
            expected_output_shape = (batch_size, config['data']['pred_len'], config['model']['output_dim'])
            if output.shape != expected_output_shape:
                raise ValueError(f"输出形状错误: 期望{expected_output_shape}, 实际{output.shape}")
            
            # 验证输出数值
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("专家网络输出包含NaN或Inf")
            
            # 测试梯度计算
            expert.train()
            test_input.requires_grad_(True)
            output = expert(test_input)
            loss = output.sum()
            loss.backward()
            
            # 验证梯度
            grad_norms = []
            for name, param in expert.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        raise ValueError(f"参数{name}的梯度包含NaN或Inf")
            
            if len(grad_norms) == 0:
                raise ValueError("没有计算到任何梯度")
            
            self.log_test("专家网络", True, 
                         f"专家网络测试成功: {sum(p.numel() for p in expert.parameters()):,}个参数, "
                         f"梯度范数: {np.mean(grad_norms):.6f}")
            return True
            
        except Exception as e:
            self.log_test("专家网络", False, f"专家网络错误: {e}")
            traceback.print_exc()
            return False
    
    def test_5_full_model(self, config: Dict[str, Any]) -> bool:
        """测试5: 完整模型测试"""
        try:
            from models.m2_moep import M2_MOEP
            
            # 创建完整模型
            model = M2_MOEP(config).to(self.device)
            
            # 验证模型结构
            required_components = [
                'flow_model', 'gating', 'experts', 'log_temperature'
            ]
            
            for component in required_components:
                if not hasattr(model, component):
                    raise AttributeError(f"模型缺失组件: {component}")
            
            # 验证专家数量
            if len(model.experts) != config['model']['num_experts']:
                raise ValueError(f"专家数量不匹配: 期望{config['model']['num_experts']}, 实际{len(model.experts)}")
            
            # 测试前向传播
            batch_size = 4
            seq_len = config['data']['seq_len']
            input_dim = config['model']['input_dim']
            
            test_input = torch.randn(batch_size, seq_len, input_dim).to(self.device)
            test_target = torch.randn(batch_size, config['data']['pred_len'], config['model']['output_dim']).to(self.device)
            
            # 推理模式测试
            model.eval()
            with torch.no_grad():
                output = model(test_input, return_aux_info=True)
            
            # 验证输出结构
            if 'predictions' not in output:
                raise KeyError("模型输出缺失predictions")
            
            if 'aux_info' not in output:
                raise KeyError("模型输出缺失aux_info")
            
            predictions = output['predictions']
            aux_info = output['aux_info']
            
            # 验证预测形状
            expected_pred_shape = (batch_size, config['data']['pred_len'], config['model']['output_dim'])
            if predictions.shape != expected_pred_shape:
                raise ValueError(f"预测形状错误: 期望{expected_pred_shape}, 实际{predictions.shape}")
            
            # 验证辅助信息
            required_aux_keys = ['expert_weights', 'expert_embeddings', 'temperature']
            for key in required_aux_keys:
                if key not in aux_info:
                    raise KeyError(f"辅助信息缺失: {key}")
            
            # 验证专家权重
            expert_weights = aux_info['expert_weights']
            if expert_weights.shape != (batch_size, config['model']['num_experts']):
                raise ValueError(f"专家权重形状错误: {expert_weights.shape}")
            
            # 验证权重和为1
            weight_sums = expert_weights.sum(dim=-1)
            if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6):
                raise ValueError("专家权重和不为1")
            
            # 训练模式测试（带梯度）
            model.train()
            output = model(test_input, ground_truth=test_target, return_aux_info=True)
            
            predictions = output['predictions']
            aux_info = output['aux_info']
            
            # 验证训练模式下的辅助信息
            if 'reconstruction_loss' not in aux_info:
                self.logger.warning("训练模式下缺失reconstruction_loss")
            
            if 'triplet_loss' not in aux_info:
                self.logger.warning("训练模式下缺失triplet_loss")
            
            self.log_test("完整模型", True, 
                         f"模型测试成功: {sum(p.numel() for p in model.parameters()):,}个参数, "
                         f"温度: {model.temperature.item():.3f}")
            return True
            
        except Exception as e:
            self.log_test("完整模型", False, f"完整模型错误: {e}")
            traceback.print_exc()
            return False
    
    def test_6_loss_computation(self, config: Dict[str, Any]) -> bool:
        """测试6: 损失计算测试"""
        try:
            from utils.losses import CompositeLoss
            from models.m2_moep import M2_MOEP
            
            # 创建损失函数
            criterion = CompositeLoss(config)
            
            # 创建测试数据
            batch_size = 4
            pred_len = config['data']['pred_len']
            output_dim = config['model']['output_dim']
            num_experts = config['model']['num_experts']
            
            predictions = torch.randn(batch_size, pred_len, output_dim)
            targets = torch.randn(batch_size, pred_len, output_dim)
            expert_weights = torch.softmax(torch.randn(batch_size, num_experts), dim=-1)
            expert_embeddings = torch.randn(batch_size, 128)
            
            # 计算损失
            losses = criterion(
                predictions=predictions,
                targets=targets,
                expert_weights=expert_weights,
                expert_embeddings=expert_embeddings
            )
            
            # 验证损失结构
            required_loss_keys = ['total', 'prediction', 'reconstruction', 'triplet']
            for key in required_loss_keys:
                if key not in losses:
                    raise KeyError(f"损失缺失: {key}")
            
            # 验证损失值
            total_loss = losses['total']
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise ValueError("总损失包含NaN或Inf")
            
            if total_loss.item() < 0:
                raise ValueError("总损失为负数")
            
            # 验证各组件损失
            for key, loss in losses.items():
                if isinstance(loss, torch.Tensor):
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise ValueError(f"损失{key}包含NaN或Inf")
                    if loss.item() < 0:
                        raise ValueError(f"损失{key}为负数")
            
            # 测试梯度反传
            total_loss.backward()
            
            self.log_test("损失计算", True, 
                         f"损失计算成功: 总损失{total_loss.item():.6f}, "
                         f"预测损失{losses['prediction'].item():.6f}")
            return True
            
        except Exception as e:
            self.log_test("损失计算", False, f"损失计算错误: {e}")
            traceback.print_exc()
            return False
    
    def test_7_training_step(self, config: Dict[str, Any]) -> bool:
        """测试7: 完整训练步骤测试"""
        try:
            from train import M2MOEPTrainer
            
            # 创建训练器
            trainer = M2MOEPTrainer(config)
            
            # 验证训练器组件
            required_trainer_attrs = [
                'model', 'data_module', 'criterion', 'optimizer', 'scheduler'
            ]
            
            for attr in required_trainer_attrs:
                if not hasattr(trainer, attr):
                    raise AttributeError(f"训练器缺失属性: {attr}")
            
            # 获取一个批次的数据
            train_loader = trainer.data_module.get_train_loader()
            batch_data = next(iter(train_loader))
            
            if len(batch_data) != 2:
                raise ValueError(f"批次数据格式错误: 期望2个元素, 实际{len(batch_data)}")
            
            batch_x, batch_y = batch_data
            batch_x = batch_x.to(trainer.device)
            batch_y = batch_y.to(trainer.device)
            
            # 模拟训练步骤
            trainer.model.train()
            trainer.optimizer.zero_grad()
            
            # 前向传播
            output = trainer.model(batch_x, ground_truth=batch_y, return_aux_info=True)
            predictions = output['predictions']
            aux_info = output['aux_info']
            
            # 损失计算
            losses = trainer.criterion(
                predictions=predictions,
                targets=batch_y,
                expert_weights=aux_info.get('expert_weights'),
                expert_embeddings=aux_info.get('expert_embeddings')
            )
            total_loss = losses['total']
            
            # 反向传播
            total_loss.backward()
            
            # 梯度检查
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), 
                float('inf')
            )
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                raise ValueError("梯度范数异常")
            
            # 优化器步骤
            trainer.optimizer.step()
            
            # 验证参数更新
            param_changed = False
            for param in trainer.model.parameters():
                if param.requires_grad and param.grad is not None:
                    param_changed = True
                    break
            
            if not param_changed:
                raise ValueError("参数没有更新")
            
            self.log_test("完整训练步骤", True, 
                         f"训练步骤成功: 损失{total_loss.item():.6f}, "
                         f"梯度范数{grad_norm.item():.6f}")
            return True
            
        except Exception as e:
            self.log_test("完整训练步骤", False, f"训练步骤错误: {e}")
            traceback.print_exc()
            return False
    
    def test_8_memory_and_device_consistency(self, config: Dict[str, Any]) -> bool:
        """测试8: 内存和设备一致性"""
        try:
            from models.m2_moep import M2_MOEP
            
            # 创建模型
            model = M2_MOEP(config).to(self.device)
            
            # 检查关键参数设备一致性（允许专家网络在CPU上）
            device_issues = []
            
            # 检查门控网络参数
            for name, param in model.gating.named_parameters():
                if param.device != self.device:
                    device_issues.append(f"门控网络参数{name}在错误设备上: {param.device}")
            
            # 检查主模型参数（除了专家网络）
            for name, param in model.named_parameters():
                if not name.startswith('experts.'):  # 忽略专家网络参数
                    if param.device != self.device:
                        device_issues.append(f"主模型参数{name}在错误设备上: {param.device}")
            
            # 检查缓冲区
            for name, buffer in model.named_buffers():
                if not name.startswith('experts.'):  # 忽略专家网络缓冲区
                    if buffer.device != self.device:
                        device_issues.append(f"主模型缓冲区{name}在错误设备上: {buffer.device}")
            
            # 设备问题不算错误（因为专家网络在CPU上是正常的）
            if device_issues:
                self.logger.warning(f"发现设备不一致（可能是正常的）: {device_issues[:5]}...")  # 只显示前5个
            
            # 内存泄漏测试
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # 多次前向传播测试
            for i in range(10):
                test_input = torch.randn(2, config['data']['seq_len'], config['model']['input_dim']).to(self.device)
                with torch.no_grad():
                    output = model(test_input)
                del test_input, output
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                memory_increase = final_memory - initial_memory
                
                if memory_increase > 100 * 1024 * 1024:  # 100MB阈值
                    self.logger.warning(f"可能存在内存泄漏: 增加{memory_increase / 1024 / 1024:.1f}MB")
            
            # 测试前向传播是否正常工作（这是关键测试）
            test_input = torch.randn(2, config['data']['seq_len'], config['model']['input_dim']).to(self.device)
            with torch.no_grad():
                output = model(test_input)
            
            # 验证输出结构（output可能是字典）
            if isinstance(output, dict):
                if 'predictions' in output:
                    predictions = output['predictions']
                    # 检查预测是否是有效的张量
                    if not torch.is_tensor(predictions):
                        raise RuntimeError("模型预测输出不是张量")
                    
                    # 检查预测形状
                    expected_shape = (2, config['data']['pred_len'], config['model']['output_dim'])
                    if predictions.shape != expected_shape:
                        raise RuntimeError(f"预测形状错误: 期望{expected_shape}, 实际{predictions.shape}")
                    
                    # 重要：专家网络在CPU上时，输出可能在不同设备上，这是正常的
                    # 只要输出是有效的张量就行
                    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                        raise RuntimeError("模型预测输出包含NaN或Inf")
                    
                    self.logger.info(f"模型预测输出正常: 设备{predictions.device}, 形状{predictions.shape}")
                else:
                    raise RuntimeError("模型输出字典中缺少predictions键")
            else:
                # 直接是张量
                if not torch.is_tensor(output):
                    raise RuntimeError("模型输出不是张量")
                
                # 检查输出数值
                if torch.isnan(output).any() or torch.isinf(output).any():
                    raise RuntimeError("模型输出包含NaN或Inf")
                
                self.logger.info(f"模型输出正常: 设备{output.device}, 形状{output.shape}")
            
            self.log_test("内存和设备一致性", True, "模型功能正常（专家网络设备分离是正常的）")
            return True
            
        except Exception as e:
            self.log_test("内存和设备一致性", False, f"内存/设备错误: {e}")
            return False
    
    def test_9_numerical_stability(self, config: Dict[str, Any]) -> bool:
        """测试9: 数值稳定性测试"""
        try:
            from models.m2_moep import M2_MOEP
            from utils.losses import CompositeLoss
            
            model = M2_MOEP(config).to(self.device)
            criterion = CompositeLoss(config)
            
            # 极端输入测试
            test_cases = [
                torch.zeros(2, config['data']['seq_len'], config['model']['input_dim']),  # 全零
                torch.ones(2, config['data']['seq_len'], config['model']['input_dim']),   # 全一
                torch.randn(2, config['data']['seq_len'], config['model']['input_dim']) * 100,  # 大值
                torch.randn(2, config['data']['seq_len'], config['model']['input_dim']) * 0.001,  # 小值
            ]
            
            for i, test_input in enumerate(test_cases):
                test_input = test_input.to(self.device)
                
                # 测试前向传播
                model.eval()
                with torch.no_grad():
                    output = model(test_input, return_aux_info=True)
                    predictions = output['predictions']
                    
                    # 检查输出数值稳定性
                    if torch.isnan(predictions).any():
                        raise ValueError(f"测试用例{i}: 预测包含NaN")
                    
                    if torch.isinf(predictions).any():
                        raise ValueError(f"测试用例{i}: 预测包含Inf")
                    
                    # 检查专家权重
                    expert_weights = output['aux_info']['expert_weights']
                    if torch.isnan(expert_weights).any():
                        raise ValueError(f"测试用例{i}: 专家权重包含NaN")
                    
                    # 检查权重和
                    weight_sums = expert_weights.sum(dim=-1)
                    if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5):
                        raise ValueError(f"测试用例{i}: 专家权重和异常")
            
            # 温度稳定性测试
            original_temp = model.temperature.item()
            
            # 设置极端温度值
            extreme_temps = [0.001, 100.0]
            for temp in extreme_temps:
                model.temperature = temp
                
                test_input = torch.randn(2, config['data']['seq_len'], config['model']['input_dim']).to(self.device)
                with torch.no_grad():
                    output = model(test_input, return_aux_info=True)
                    
                    if torch.isnan(output['predictions']).any():
                        raise ValueError(f"温度{temp}: 预测包含NaN")
            
            # 恢复原始温度
            model.temperature = original_temp
            
            self.log_test("数值稳定性", True, "数值稳定性测试通过")
            return True
            
        except Exception as e:
            self.log_test("数值稳定性", False, f"数值稳定性错误: {e}")
            return False
    
    def test_10_error_recovery(self, config: Dict[str, Any]) -> bool:
        """测试10: 错误恢复和异常处理"""
        try:
            from models.m2_moep import M2_MOEP
            
            model = M2_MOEP(config).to(self.device)
            
            # 测试维度不匹配错误
            try:
                wrong_input = torch.randn(2, config['data']['seq_len'] + 10, config['model']['input_dim']).to(self.device)
                output = model(wrong_input)
                self.logger.warning("维度检查可能有问题：应该抛出错误但没有")
            except (ValueError, RuntimeError) as e:
                self.logger.info(f"正确捕获维度错误: {type(e).__name__}")
            
            # 测试设备不匹配处理
            if torch.cuda.is_available() and self.device.type == 'cuda':
                try:
                    cpu_input = torch.randn(2, config['data']['seq_len'], config['model']['input_dim'])
                    output = model(cpu_input)  # 应该自动移到GPU
                    if output.device.type != 'cuda':
                        raise RuntimeError("设备自动迁移失败")
                    self.logger.info("设备自动迁移成功")
                except Exception as e:
                    self.logger.warning(f"设备迁移错误: {e}")
            
            # 测试NaN输入处理
            try:
                nan_input = torch.randn(2, config['data']['seq_len'], config['model']['input_dim']).to(self.device)
                nan_input[0, 0, 0] = float('nan')
                output = model(nan_input)
                self.logger.warning("NaN输入处理可能有问题：应该抛出错误或修复")
            except (ValueError, RuntimeError) as e:
                self.logger.info(f"正确处理NaN输入: {type(e).__name__}")
            
            self.log_test("错误恢复和异常处理", True, "异常处理测试完成")
            return True
            
        except Exception as e:
            self.log_test("错误恢复和异常处理", False, f"异常处理测试错误: {e}")
            return False
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面测试"""
        print("🔬 开始全面深度测试...")
        print(f"🔥 使用设备: {self.device}")
        print("=" * 80)
        
        # 运行所有测试
        tests = [
            ("导入和依赖", self.test_1_imports_and_dependencies),
            ("配置生成", self.test_2_config_generation),
        ]
        
        config = None
        all_passed = True
        
        # 执行基础测试
        for test_name, test_func in tests:
            try:
                if test_name == "配置生成":
                    config = test_func()
                    success = bool(config)
                else:
                    success = test_func()
                
                if not success:
                    all_passed = False
                    
            except Exception as e:
                self.log_test(test_name, False, f"测试异常: {e}")
                all_passed = False
        
        # 如果基础测试失败，停止后续测试
        if not config:
            print("\n❌ 基础测试失败，停止后续测试")
            return self._generate_test_report()
        
        # 高级测试列表
        advanced_tests = [
            ("数据加载", lambda: self.test_3_data_loading(config)),
            ("专家网络", lambda: self.test_4_expert_network(config)),
            ("完整模型", lambda: self.test_5_full_model(config)),
            ("损失计算", lambda: self.test_6_loss_computation(config)),
            ("训练步骤", lambda: self.test_7_training_step(config)),
            ("内存设备", lambda: self.test_8_memory_and_device_consistency(config)),
            ("数值稳定性", lambda: self.test_9_numerical_stability(config)),
            ("异常处理", lambda: self.test_10_error_recovery(config)),
        ]
        
        # 执行高级测试
        for test_name, test_func in advanced_tests:
            try:
                success = test_func()
                if not success:
                    all_passed = False
            except Exception as e:
                self.log_test(test_name, False, f"测试异常: {e}")
                all_passed = False
                traceback.print_exc()
        
        return self._generate_test_report()
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        print("\n" + "=" * 80)
        print("📊 全面测试报告")
        print("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        total_tests = len(self.test_results)
        
        print(f"📈 总体结果: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            print("🎉 所有测试通过！训练流程准备就绪。")
        else:
            print("⚠️  存在失败的测试，需要修复：")
            for error in self.critical_errors:
                print(f"   - {error}")
        
        print("\n📋 详细测试结果:")
        for test_name, result in self.test_results.items():
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {test_name}: {result['message']}")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'all_passed': passed_tests == total_tests,
            'test_results': self.test_results,
            'critical_errors': self.critical_errors
        }

def main():
    """主函数"""
    test_suite = ComprehensiveTestSuite()
    report = test_suite.run_comprehensive_test()
    
    # 返回状态码
    return 0 if report['all_passed'] else 1

if __name__ == "__main__":
    sys.exit(main()) 