import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, using LSTM as fallback")

class FFTmsMambaExpert(nn.Module):
    """
    FFT + ms-Mamba 专家网络
    早期融合策略：原始时间序列与FFT频域表示拼接
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 从配置中获取参数
        self.d_model = config['model']['expert_params'].get('mamba_d_model', 256)
        self.mamba_d_model = self.d_model  # 保持一致性
        
        # 输入和输出维度
        self.input_dim = config['model']['input_dim']
        self.output_dim = config['model']['output_dim']
        self.seq_len = config['model']['seq_len']
        self.pred_len = config['model']['pred_len']
        
        # 初始尺度列表，可训练 Δ
        init_scales = config['model']['expert_params'].get('mamba_scales', [1, 2, 4])
        self.learnable_deltas = nn.Parameter(torch.tensor(init_scales, dtype=torch.float))
        
        # 专家ID（用于个性化）
        self.expert_id = config['model'].get('current_expert_id', 0)
        
        # 1. 输入投影层
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # 2. FFT融合层
        self.fft_fusion = nn.Linear(self.d_model, self.d_model)  # 修复：D->D而非D*3->D
        
        # 3. 多尺度Mamba层
        self.multi_scale_mamba = nn.ModuleList()
        
        # 检查是否有CUDA和mamba-ssm
        self.use_mamba = self._check_mamba_availability()
        
        for scale in init_scales:
            if self.use_mamba:
                try:
                    from mamba_ssm import Mamba
                    mamba_layer = Mamba(
                        d_model=self.d_model,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    )
                    self.multi_scale_mamba.append(mamba_layer)
                    print(f"✅ 成功初始化Mamba层 (scale={scale})")
                except Exception as e:
                    print(f"❌ Mamba层初始化失败 (scale={scale}): {e}")
                    self.use_mamba = False
                    print("切换到LSTM替代方案")
                    # 重新初始化所有层为LSTM
                    self.multi_scale_mamba = nn.ModuleList()
                    for _ in init_scales:
                        self.multi_scale_mamba.append(
                            nn.LSTM(self.d_model, self.d_model, batch_first=True)
                        )
                    break
            else:
                # 使用LSTM作为替代
                self.multi_scale_mamba.append(
                    nn.LSTM(self.d_model, self.d_model, batch_first=True)
                )
                print(f"🔄 使用LSTM替代Mamba (scale={scale})")
        
        # 4. 尺度融合层
        self.scale_fusion = nn.Linear(self.d_model * len(init_scales), self.d_model)
        
        # 5. 输出投影层
        self.output_projection = nn.Linear(self.d_model, self.output_dim)
        
        # 6. 预测头
        self.prediction_head = nn.Linear(self.seq_len, self.pred_len)
        
        # 7. 专家个性化层
        self.expert_personalization = nn.Linear(self.d_model, self.d_model)
        
        # 8. Dropout
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重
        self._initialize_weights()
        
        # 报告Mamba使用状态
        self._report_mamba_status()

    def _report_mamba_status(self):
        """报告Mamba的使用状态"""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.use_mamba:
            print(f"🚀 专家{self.expert_id}成功使用Mamba架构")
            print(f"   - 尺度数量: {len(self.multi_scale_mamba)}")
            print(f"   - d_model: {self.d_model}")
            print(f"   - 设备: {device}")
        else:
            print(f"⚠️  专家{self.expert_id}使用LSTM替代Mamba")
            print(f"   - 尺度数量: {len(self.multi_scale_mamba)}")
            print(f"   - d_model: {self.d_model}")
            print(f"   - 设备: {device}")

    def _initialize_weights(self):
        """改进的权重初始化，专门针对时间序列预测优化 - 🔧 超保守版本"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # 🔧 关键修复：使用更保守的初始化策略
                if 'fft_fusion' in name:
                    # FFT融合层使用极小的正交初始化
                    nn.init.orthogonal_(m.weight, gain=0.01)  # 从0.1减少到0.01
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif 'input_projection' in name:
                    # 输入投影层使用更保守的初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.1)  # 从默认1.0减少到0.1
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif 'output_projection' in name or 'prediction_head' in name:
                    # 输出层使用极小方差的正态分布初始化
                    nn.init.normal_(m.weight, mean=0.0, std=0.001)  # 从0.01减少到0.001
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif 'scale_fusion' in name:
                    # 尺度融合层使用极保守的Xavier初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.1)  # 从0.5减少到0.1
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    # 其他线性层使用极保守的Xavier初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.01)  # 从0.5减少到0.01
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.LSTM):
                # LSTM权重初始化 - 更保守
                for param_name, param in m.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param, gain=0.1)  # 从0.5减少到0.1
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param, gain=0.1)  # 从0.5减少到0.1
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
                        # LSTM遗忘门偏置设为较小值
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(0.5)  # 从1.0减少到0.5
        
        # 特殊处理：可学习的尺度参数 - 更保守
        if hasattr(self, 'learnable_deltas'):
            # 🔧 关键修复：初始化为更小的值，避免极端的上下采样
            with torch.no_grad():
                self.learnable_deltas.data = torch.clamp(self.learnable_deltas.data, 1.0, 2.0)  # 从3.0减少到2.0
        
        print(f"✅ 专家{self.expert_id}权重初始化完成 (超保守策略)")

    def _stable_fft_fusion(self, x: torch.Tensor) -> torch.Tensor:
        """
        数值稳定的FFT融合 - 完全修复版本
        :param x: 输入张量 [B, T, D]
        :return: 融合后的张量 [B, T, D]
        """
        try:
            # 输入检查和修复
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("警告: FFT融合输入包含NaN或Inf，执行修复")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 输入归一化 - 使用更保守的归一化
            x_normalized = F.layer_norm(x, x.shape[-1:])
            
            # 计算FFT - 仅在时间维度上进行
            x_fft = torch.fft.fft(x_normalized, dim=-2)
            
            # 获取幅度谱和相位谱
            magnitude = torch.abs(x_fft)
            phase = torch.angle(x_fft)
            
            # 幅度谱稳定化处理 - 更保守的范围
            magnitude = torch.clamp(magnitude, min=1e-6, max=5.0)
            
            # 对数归一化防止数值过大 - 更保守的范围
            log_magnitude = torch.log(magnitude + 1e-6)
            log_magnitude = torch.clamp(log_magnitude, min=-3.0, max=3.0)
            
            # 相位归一化 - 限制在合理范围内
            phase = torch.clamp(phase, min=-np.pi, max=np.pi)
            
            # 🔧 关键修复：不再拼接特征，而是使用加权融合
            if hasattr(self, 'fft_fusion') and self.fft_fusion is not None:
                # 创建加权特征而非拼接特征
                # 使用门控机制控制每个特征的贡献
                alpha = 0.3  # 幅度特征权重
                beta = 0.2   # 相位特征权重
                gamma = 0.5  # 原始特征权重
                
                # 确保权重和为1
                total_weight = alpha + beta + gamma
                alpha, beta, gamma = alpha/total_weight, beta/total_weight, gamma/total_weight
                
                # 将频域特征转换回时域并融合
                magnitude_features = F.layer_norm(log_magnitude, log_magnitude.shape[-1:])
                phase_features = F.layer_norm(phase, phase.shape[-1:])
                
                # 加权融合而非拼接
                fused_features = (
                    gamma * x_normalized + 
                    alpha * magnitude_features + 
                    beta * phase_features
                )
                
                # 通过fusion layer进行额外的特征变换
                # 🔧 关键修复：输入维度保持D而非D*3
                fusion_input = fused_features  # [B, T, D]
                fused_x = self.fft_fusion(fusion_input)  # [B, T, D]
            else:
                # 简单处理：直接使用原始特征
                fused_x = x_normalized
            
            # 输出检查和修复
            if torch.isnan(fused_x).any() or torch.isinf(fused_x).any():
                print("警告: FFT融合输出包含异常值，使用原始输入")
                return x
            
            # 残差连接（更保守的权重）
            residual_weight = 0.1  # 更小的融合权重
            output = (1 - residual_weight) * x + residual_weight * fused_x
            
            # 最终数值稳定性保证
            output = torch.clamp(output, min=-5.0, max=5.0)
            
            return output
            
        except Exception as e:
            print(f"错误: FFT融合失败: {e}")
            return x  # 出错时返回原始输入

    def forward(self, x, return_features=True):
        """
        FFT+ms-Mamba专家前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            return_features: 是否返回特征（True）还是预测结果（False）
        Returns:
            如果return_features=True: 特征张量 [batch_size, seq_len, hidden_dim]
            如果return_features=False: 预测张量 [batch_size, pred_len, output_dim]
        """
        # 设备一致性检查
        model_device = next(self.parameters()).device
        if x.device != model_device:
            print(f"⚠️  设备不匹配: 输入在{x.device}, 模型在{model_device}")
            x = x.to(model_device)
            print(f"✅ 输入已移动到{model_device}")
        
        batch_size, seq_len, input_dim = x.shape
        
        # 验证输入维度
        if input_dim != self.input_dim:
            raise ValueError(f"输入维度不匹配: 期望{self.input_dim}, 实际{input_dim}")
        
        if seq_len != self.seq_len:
            raise ValueError(f"序列长度不匹配: 期望{self.seq_len}, 实际{seq_len}")
        
        # === 1. 输入投影 ===
        x_proj = self.input_projection(x)  # [B, T, d_model]
        
        # === 2. FFT融合（可选，数值稳定版本） ===
        if hasattr(self, 'fft_fusion') and self.fft_fusion is not None:
            x_proj = self._stable_fft_fusion(x_proj)
        
        # === 3. 多尺度处理 ===
        scale_outputs = []
        
        for i, mamba_layer in enumerate(self.multi_scale_mamba):
            # 使用可学习的delta参数并添加数值稳定性检查
            delta = torch.clamp(self.learnable_deltas[i], min=1.0, max=8.0)
            scale = delta.item()  # 转换为标量
            
            # 获取当前尺度的输入
            if scale == 1:
                scaled_input = x_proj
            elif scale > 1:
                # 下采样
                scaled_input = self._downsample(x_proj, scale)
            else:
                # 上采样
                target_len = int(seq_len / scale)
                scaled_input = self._upsample(x_proj, target_len)
            
            # 确保设备一致性
            if scaled_input.device != model_device:
                scaled_input = scaled_input.to(model_device)
            
            # 通过Mamba层或LSTM层
            if self.use_mamba:
                scaled_output = mamba_layer(scaled_input)
            else:
                # LSTM需要特殊处理
                scaled_output, _ = mamba_layer(scaled_input)
            
            # 恢复到原始序列长度
            if scaled_output.size(1) != seq_len:
                if scaled_output.size(1) < seq_len:
                    # 需要上采样
                    scaled_output = self._upsample(scaled_output, seq_len)
                else:
                    # 需要下采样
                    scale_factor = scaled_output.size(1) / seq_len
                    scaled_output = self._downsample(scaled_output, scale_factor)
            
            scale_outputs.append(scaled_output)
        
        # === 4. 尺度融合 ===
        if len(scale_outputs) > 1:
            fused_features = torch.cat(scale_outputs, dim=-1)  # [B, T, d_model * num_scales]
            fused_output = self.scale_fusion(fused_features)   # [B, T, d_model]
        else:
            fused_output = scale_outputs[0]
        
        # === 5. 专家个性化 ===
        personalized_output = self.expert_personalization(fused_output)
        personalized_output = self.dropout(personalized_output)
        
        # 如果只需要特征，直接返回
        if return_features:
            return personalized_output  # [B, T, d_model] 或 [B, T, hidden_dim]
        
        # === 6. 输出投影 ===
        output_features = self.output_projection(personalized_output)  # [B, T, output_dim]
        
        # === 7. 时序预测 ===
        # 转置以进行时序变换: [B, output_dim, T]
        output_transposed = output_features.transpose(1, 2)
        
        # 预测头: [B, output_dim, T] -> [B, output_dim, pred_len]
        predictions_transposed = self.prediction_head(output_transposed)
        
        # 转回: [B, pred_len, output_dim]
        predictions = predictions_transposed.transpose(1, 2)
        
        # 最终设备检查
        if predictions.device != model_device:
            predictions = predictions.to(model_device)
        
        return predictions

    def _downsample(self, x, scale):
        """
        下采样到指定尺度
        :param x: 输入序列 [B, seq_len, d_model]
        :param scale: 下采样尺度
        :return: 下采样后的序列
        """
        if scale == 1:
            return x
        
        # 确保scale是整数
        scale = int(scale)
        
        # 使用平均池化下采样
        x_permuted = x.permute(0, 2, 1)  # [B, d_model, seq_len]
        downsampled = F.avg_pool1d(x_permuted, kernel_size=scale, stride=scale)
        return downsampled.permute(0, 2, 1)  # [B, new_seq_len, d_model]

    def _upsample(self, x, target_len):
        """
        上采样到目标长度
        :param x: 输入序列 [B, seq_len, d_model]
        :param target_len: 目标长度
        :return: 上采样后的序列
        """
        if x.size(1) == target_len:
            return x
        
        # 使用线性插值上采样，兼容不同PyTorch版本
        x_permuted = x.permute(0, 2, 1)  # [B, d_model, seq_len]
        
        # 检查PyTorch版本兼容性
        try:
            # 尝试使用linear模式（PyTorch >= 1.9推荐）
            upsampled = F.interpolate(x_permuted, size=target_len, mode='linear', align_corners=False)
        except (RuntimeError, ValueError):
            # 回退到nearest模式（兼容所有版本）
            print("警告: linear插值不支持，使用nearest模式")
            upsampled = F.interpolate(x_permuted, size=target_len, mode='nearest')
        
        return upsampled.permute(0, 2, 1)  # [B, target_len, d_model]

    def _check_mamba_availability(self):
        """检查Mamba是否可用，并进行设备兼容性测试"""
        # 获取当前设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            import mamba_ssm
            
            # 如果没有CUDA，直接返回False
            if not torch.cuda.is_available():
                print("⚠️  Mamba需要CUDA，但CUDA不可用，使用LSTM替代")
                return False
            
            print(f"在设备 {device} 上测试Mamba...")
            
            # 设备兼容性测试
            test_tensor = torch.randn(1, 10, 64, device=device)
            from mamba_ssm import Mamba
            test_layer = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to(device)
            _ = test_layer(test_tensor)
            
            print(f"✅ Mamba在设备 {device} 上测试成功")
            return True
        except Exception as e:
            print(f"⚠️  Mamba在设备 {device} 上不可用: {e}")
            return False

    def to(self, device):
        """重写to方法，确保所有组件都移动到正确设备"""
        # 如果使用mamba且设备不是CUDA，强制使用CUDA
        if self.use_mamba and device.type != 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"使用mamba时强制切换到CUDA设备")
            else:
                print(f"⚠️mamba需要CUDA但CUDA不可用，将切换到LSTM")
                self.use_mamba = False
                # 重新初始化为LSTM
                self.multi_scale_mamba = nn.ModuleList()
                for _ in range(len(self.learnable_deltas)):
                    self.multi_scale_mamba.append(
                        nn.LSTM(self.d_model, self.d_model, batch_first=True)
                    )
        
        super().to(device)
        
        # 确保可学习参数在正确设备上
        if hasattr(self, 'learnable_deltas'):
            self.learnable_deltas.data = self.learnable_deltas.data.to(device)
        
        # 确保所有层都在正确设备上
        self.input_projection.to(device)
        if hasattr(self, 'fft_fusion') and self.fft_fusion is not None:
            self.fft_fusion.to(device)
        
        # 确保多尺度mamba层在正确设备上
        for layer in self.multi_scale_mamba:
            layer.to(device)
        
        self.scale_fusion.to(device)
        self.output_projection.to(device)
        self.prediction_head.to(device)
        self.expert_personalization.to(device)
        
        return self