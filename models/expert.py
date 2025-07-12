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

class FrequencyMambaFFTExtractor(nn.Module):
    """
    借鉴FrequencyMamba的FFT特征提取器
    提取幅度和相位信息，使用可学习的频率选择
    """
    def __init__(self, seq_len, input_dim, fft_bins=16):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.fft_bins = min(fft_bins, seq_len // 2 + 1)  # 确保不超过FFT输出长度
        
        # 可学习的频率选择权重
        self.freq_attention = nn.Parameter(torch.ones(self.fft_bins) / self.fft_bins)
        
        # 数值稳定性参数
        self.magnitude_scale = nn.Parameter(torch.tensor(1.0))
        self.phase_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            fft_features: [batch_size, input_dim * fft_bins * 2]
        """
        try:
            batch_size = x.size(0)
            
            # 数值预处理：去均值和窗函数
            x_processed = x - x.mean(dim=1, keepdim=True)
            
            # 应用汉宁窗减少频谱泄漏
            window = torch.hann_window(self.seq_len, device=x.device)
            x_windowed = x_processed * window.unsqueeze(0).unsqueeze(-1)
            
            # 向量化实现：交换维度以对每个变量的seq_len进行FFT
            x_permuted = x_windowed.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]

            # 批量进行FFT
            fft_result = torch.fft.rfft(x_permuted, dim=2)  # [batch_size, input_dim, seq_len//2 + 1]

            # 批量提取幅度和相位，添加更好的数值稳定性
            magnitudes = torch.abs(fft_result)
            phases = torch.angle(fft_result)

            # 改进的数值稳定性处理
            magnitudes = torch.clamp(magnitudes, min=1e-8, max=50.0)  # 降低上限
            phases = torch.clamp(phases, min=-np.pi, max=np.pi)

            # 选择前N个频率分量
            if magnitudes.size(2) >= self.fft_bins:
                selected_magnitudes = magnitudes[:, :, :self.fft_bins]
                selected_phases = phases[:, :, :self.fft_bins]
            else:
                # 如果序列太短，进行零填充
                pad_width = self.fft_bins - magnitudes.size(2)
                selected_magnitudes = F.pad(magnitudes, (0, pad_width))
                selected_phases = F.pad(phases, (0, pad_width))
            
            # 应用可学习的频率权重
            freq_weights = F.softmax(self.freq_attention, dim=0).view(1, 1, -1)
            weighted_magnitudes = selected_magnitudes * freq_weights
            weighted_phases = selected_phases * torch.clamp(self.phase_scale, 0.01, 1.0)

            # 组合幅度和相位信息
            combined_features = torch.cat([
                weighted_magnitudes * torch.clamp(self.magnitude_scale, 0.1, 10.0),
                weighted_phases
            ], dim=2)

            # 拼接所有变量的FFT特征
            all_fft_features = combined_features.view(batch_size, -1)

            # 最终数值稳定性检查
            if torch.isnan(all_fft_features).any() or torch.isinf(all_fft_features).any():
                print("警告: FFT特征包含NaN/Inf，使用备用方案")
                # 使用简单的时域统计特征作为备用
                mean_features = x.mean(dim=1)  # [batch_size, input_dim]
                std_features = x.std(dim=1)   # [batch_size, input_dim]
                backup_features = torch.cat([mean_features, std_features], dim=1)
                # 扩展到目标维度
                target_dim = self.input_dim * self.fft_bins * 2
                if backup_features.size(1) < target_dim:
                    pad_size = target_dim - backup_features.size(1)
                    backup_features = F.pad(backup_features, (0, pad_size))
                else:
                    backup_features = backup_features[:, :target_dim]
                return backup_features

            return all_fft_features

        except Exception as e:
            print(f"FFT特征提取失败: {e}")
            # 返回时域统计特征作为备用
            mean_features = x.mean(dim=1)
            std_features = x.std(dim=1)
            backup_features = torch.cat([mean_features, std_features], dim=1)
            target_dim = self.input_dim * self.fft_bins * 2
            if backup_features.size(1) < target_dim:
                pad_size = target_dim - backup_features.size(1)
                backup_features = F.pad(backup_features, (0, pad_size))
            else:
                backup_features = backup_features[:, :target_dim]
            return backup_features

class FrequencyMambaGate(nn.Module):
    """
    借鉴FrequencyMamba的门控机制
    控制时域和频域特征的融合
    """
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Sigmoid()
        )
        
    def forward(self, time_features, freq_features):
        """
        Args:
            time_features: [batch_size, seq_len, d_model]
            freq_features: [batch_size, seq_len, d_model]
        Returns:
            gated_features: [batch_size, seq_len, d_model]
        """
        # 计算门控权重
        gate_weights = self.gate(time_features + freq_features)
        
        # 门控融合
        gated_features = gate_weights * time_features + (1 - gate_weights) * freq_features
        
        return gated_features

class FFTmsMambaExpert(nn.Module):
    """
    FFT + ms-Mamba 专家网络
    借鉴FrequencyMamba的频域处理策略
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
        
        # === FrequencyMamba风格的频域处理 ===
        # 1. FFT特征提取器
        fft_bins = config['model']['expert_params'].get('fft_bins', 16)
        self.fft_extractor = FrequencyMambaFFTExtractor(self.seq_len, self.input_dim, fft_bins)
        
        # 2. 时域投影
        self.time_projection = nn.Linear(self.input_dim, self.d_model)
        
        # 3. 频域投影
        fft_feature_size = self.input_dim * fft_bins * 2
        self.freq_projection = nn.Linear(fft_feature_size, self.d_model)
        
        # 4. 时频融合层
        self.time_freq_fusion = nn.Linear(self.d_model * 2, self.d_model)
        
        # 5. 频域门控机制
        self.freq_gate = FrequencyMambaGate(self.d_model)
        
        # === 多尺度Mamba层 ===
        self.multi_scale_mamba = nn.ModuleList()
        
        # 检查是否有CUDA和mamba-ssm
        self.use_mamba = self._check_mamba_availability()
        
        # 统一初始化策略
        if self.use_mamba:
            # 尝试初始化所有Mamba层
            mamba_success = True
            for scale in init_scales:
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
                    mamba_success = False
                    break
            
            # 如果有任何Mamba层初始化失败，回退到LSTM
            if not mamba_success:
                print("🔄 Mamba初始化失败，切换到LSTM替代方案")
                self.use_mamba = False
                self.multi_scale_mamba = nn.ModuleList()
                for scale in init_scales:
                    self.multi_scale_mamba.append(
                        nn.LSTM(self.d_model, self.d_model, batch_first=True)
                    )
                    print(f"🔄 使用LSTM替代Mamba (scale={scale})")
        else:
            # 直接使用LSTM替代
            for scale in init_scales:
                self.multi_scale_mamba.append(
                    nn.LSTM(self.d_model, self.d_model, batch_first=True)
                )
                print(f"🔄 使用LSTM替代Mamba (scale={scale})")
        
        # === 输出层 ===
        # 尺度融合层
        self.scale_fusion = nn.Linear(self.d_model * len(init_scales), self.d_model)
        
        # 输出投影层
        self.output_projection = nn.Linear(self.d_model, self.output_dim)
        
        # 预测头
        self.prediction_head = nn.Linear(self.seq_len, self.pred_len)
        
        # 专家个性化层
        self.expert_personalization = nn.Linear(self.d_model, self.d_model)
        
        # Dropout
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
            print(f"🚀 专家{self.expert_id}成功使用Mamba架构 (FrequencyMamba风格)")
            print(f"   - 尺度数量: {len(self.multi_scale_mamba)}")
            print(f"   - d_model: {self.d_model}")
            print(f"   - 设备: {device}")
        else:
            print(f"⚠️  专家{self.expert_id}使用LSTM替代Mamba (FrequencyMamba风格)")
            print(f"   - 尺度数量: {len(self.multi_scale_mamba)}")
            print(f"   - d_model: {self.d_model}")
            print(f"   - 设备: {device}")

    def _initialize_weights(self):
        """改进的权重初始化，专门针对FrequencyMamba风格的架构"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'freq_projection' in name:
                    # 频域投影层使用较小的初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif 'time_projection' in name:
                    # 时域投影层使用标准初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif 'time_freq_fusion' in name:
                    # 融合层使用保守初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.2)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif 'gate' in name:
                    # 门控层使用特殊初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.1)  # 轻微的正偏置
                else:
                    # 其他层使用标准初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.LSTM):
                # LSTM权重初始化
                for param_name, param in m.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param, gain=0.5)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
                        # LSTM遗忘门偏置设为1
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)
        
        # 特殊处理：可学习的尺度参数
        if hasattr(self, 'learnable_deltas'):
            with torch.no_grad():
                self.learnable_deltas.data = torch.clamp(self.learnable_deltas.data, 1.0, 4.0)
        
        print(f"✅ 专家{self.expert_id}权重初始化完成 (FrequencyMamba风格)")

    def forward(self, x, return_features=True):
        """
        FrequencyMamba风格的前向传播
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
            x = x.to(model_device)
        
        batch_size, seq_len, input_dim = x.shape
        
        # 验证输入维度
        if input_dim != self.input_dim:
            raise ValueError(f"输入维度不匹配: 期望{self.input_dim}, 实际{input_dim}")
        
        if seq_len != self.seq_len:
            raise ValueError(f"序列长度不匹配: 期望{self.seq_len}, 实际{seq_len}")
        
        # 1. 时域和频域特征准备
        # (batch_size, seq_len, d_model)
        x_time = self.time_projection(x)
        
        # (batch_size, fft_feature_size)
        x_freq_flat = self.fft_extractor(x)
        
        # (batch_size, d_model) -> (batch_size, 1, d_model) -> (batch_size, seq_len, d_model)
        x_freq = self.freq_projection(x_freq_flat).unsqueeze(1).expand(-1, self.seq_len, -1)
        
        # 2. 时频融合
        # 使用门控机制融合
        x_proj = self.freq_gate(x_time, x_freq)
        x_proj = self.dropout(x_proj)
        
        # 3. 多尺度处理
        scale_outputs = []
        for i, scale_delta in enumerate(self.learnable_deltas):
            # 使用abs确保尺度为正
            scale = torch.abs(scale_delta)
            
            # 下采样
            x_proj_scaled = self._downsample(x_proj, scale)
            
            # Mamba/LSTM处理
            mamba_layer = self.multi_scale_mamba[i]
            
            if self.use_mamba:
                mamba_out = mamba_layer(x_proj_scaled)
            else:
                # 修复：正确处理LSTM输出
                mamba_out, _ = mamba_layer(x_proj_scaled)
            
            # 上采样
            x_up = self._upsample(mamba_out, self.seq_len)
            scale_outputs.append(x_up)
        
        # 4. 尺度融合
        x_fused = torch.cat(scale_outputs, dim=-1)
        x_fused = self.scale_fusion(x_fused)
        x_fused = self.dropout(x_fused)
        
        # 5. 专家个性化
        x_fused = self.expert_personalization(x_fused)
        
        # 残差连接
        final_output = x_proj + x_fused
        
        if return_features:
            return self.output_projection(final_output)
        
        # 6. 最终预测
        # (batch_size, seq_len, d_model) -> (batch_size, d_model, seq_len)
        final_output_permuted = final_output.permute(0, 2, 1)
        
        # (batch_size, d_model, pred_len)
        prediction = self.prediction_head(final_output_permuted)
        
        # (batch_size, pred_len, d_model)
        prediction = prediction.permute(0, 2, 1)
        
        return self.output_projection(prediction)

    def _downsample(self, x, scale):
        """下采样到指定尺度"""
        if scale == 1:
            return x
        
        scale = int(scale)
        x_permuted = x.permute(0, 2, 1)  # [B, d_model, seq_len]
        downsampled = F.avg_pool1d(x_permuted, kernel_size=scale, stride=scale)
        return downsampled.permute(0, 2, 1)  # [B, new_seq_len, d_model]

    def _upsample(self, x, target_len):
        """上采样到目标长度"""
        if x.size(1) == target_len:
            return x
        
        x_permuted = x.permute(0, 2, 1)  # [B, d_model, seq_len]
        
        try:
            upsampled = F.interpolate(x_permuted, size=target_len, mode='linear', align_corners=False)
        except (RuntimeError, ValueError):
            upsampled = F.interpolate(x_permuted, size=target_len, mode='nearest')
        
        return upsampled.permute(0, 2, 1)  # [B, target_len, d_model]

    def _check_mamba_availability(self):
        """检查Mamba可用性"""
        if not MAMBA_AVAILABLE:
            return False
        
        # Mamba要求CUDA支持
        if not torch.cuda.is_available():
            print(f"专家{self.expert_id}: CUDA不可用，Mamba需要CUDA支持，使用LSTM替代")
            return False
        
        try:
            # 在CUDA上测试创建一个小的Mamba层
            test_mamba = Mamba(d_model=16, d_state=8).cuda()
            test_input = torch.randn(1, 10, 16).cuda()  # 修复：在CUDA上创建测试输入
            _ = test_mamba(test_input)
            print(f"专家{self.expert_id}: Mamba可用性检查通过")
            return True
        except Exception as e:
            print(f"专家{self.expert_id}: Mamba可用性检查失败: {e}")
            return False

    def to(self, device):
        """设备转移"""
        result = super().to(device)
        
        # 确保所有参数都在正确的设备上
        for param in result.parameters():
            if param.device != device:
                param.data = param.data.to(device)
        
        return result