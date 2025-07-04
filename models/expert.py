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
        self.fft_fusion = nn.Linear(self.d_model * 3, self.d_model)  # 原始+幅度+相位
        
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
                except Exception as e:
                    print(f"警告: Mamba初始化失败，使用LSTM替代: {e}")
                    self.use_mamba = False
                    self.multi_scale_mamba.append(
                        nn.LSTM(self.d_model, self.d_model, batch_first=True)
                    )
            else:
                # 使用LSTM作为替代
                self.multi_scale_mamba.append(
                    nn.LSTM(self.d_model, self.d_model, batch_first=True)
                )
        
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

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _early_fft_fusion(self, x):
        """
        早期FFT融合：将原始序列、FFT幅度谱、FFT相位谱进行融合
        Args:
            x: 输入序列 [batch_size, seq_len, d_model]
        Returns:
            fused_features: 融合后的特征 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 对每个特征维度进行FFT
        fft_result = torch.fft.fft(x, dim=1)  # [B, seq_len, d_model]
        
        # 提取幅度谱和相位谱
        amplitude = torch.abs(fft_result)  # [B, seq_len, d_model]
        phase = torch.angle(fft_result)  # [B, seq_len, d_model]
        
        # 拼接原始、幅度、相位特征
        concatenated = torch.cat([x, amplitude, phase], dim=-1)  # [B, seq_len, d_model * 3]
        
        # 通过线性层融合到原始维度
        fused_features = self.fft_fusion(concatenated)  # [B, seq_len, d_model]
        
        return fused_features

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
        Returns:
            output: 预测结果 [batch_size, pred_len, output_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 1. 输入投影
        projected_features = self.input_projection(x)  # [B, seq_len, d_model]
        
        # 2. 早期FFT融合
        fused_features = self._early_fft_fusion(projected_features)  # [B, seq_len, d_model]
        
        # 3. 多尺度Mamba处理
        scale_outputs = []
        for i, mamba_layer in enumerate(self.multi_scale_mamba):
            # 取可学习 Δ（>=1）
            delta = torch.clamp(self.learnable_deltas[i], min=1.0)
            scale_val = int(delta.item())
            
            # 下采样到对应尺度
            scaled_input = self._downsample(fused_features, scale_val)
            
            # Mamba处理
            if self.use_mamba:
                scaled_output = mamba_layer(scaled_input)
            else:
                # LSTM处理
                scaled_output, _ = mamba_layer(scaled_input)
            
            # 上采样回原始长度
            scaled_output = self._upsample(scaled_output, seq_len)
            scale_outputs.append(scaled_output)
        
        # 4. 尺度融合
        concatenated = torch.cat(scale_outputs, dim=-1)  # [B, seq_len, d_model * num_scales]
        fused_output = self.scale_fusion(concatenated)  # [B, seq_len, d_model]
        
        # 5. 专家个性化
        personalized_output = self.expert_personalization(fused_output)
        personalized_output = self.dropout(personalized_output)
        
        # 6. 输出投影
        output_features = self.output_projection(personalized_output)  # [B, seq_len, output_dim]
        
        # 7. 时间维度预测（seq_len -> pred_len）
        # 转置以便对时间维度进行线性变换
        output_features = output_features.transpose(1, 2)  # [B, output_dim, seq_len]
        predictions = self.prediction_head(output_features)  # [B, output_dim, pred_len]
        predictions = predictions.transpose(1, 2)  # [B, pred_len, output_dim]
        
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
        
        # 使用线性插值上采样
        x_permuted = x.permute(0, 2, 1)  # [B, d_model, seq_len]
        upsampled = F.interpolate(x_permuted, size=target_len, mode='linear', align_corners=False)
        return upsampled.permute(0, 2, 1)  # [B, target_len, d_model]

    def _check_mamba_availability(self):
        """检查Mamba是否可用"""
        try:
            import mamba_ssm
            # 如果没有CUDA，使用CPU模式
            return torch.cuda.is_available()
        except ImportError:
            return False