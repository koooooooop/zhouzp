import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

# === 强大的Real NVP风格Flow模型实现 ===
class RealNVPCouplingLayer(nn.Module):
    """
    Real NVP耦合层 - 更强大的Normalizing Flow构建块
    使用深层网络和更好的数值稳定性
    """
    def __init__(self, input_dim, hidden_dim, mask, num_layers=3):
        super().__init__()
        self.register_buffer('mask', mask)
        
        # 构建更深的scale和translate网络
        scale_layers = []
        translate_layers = []
        
        # 输入层
        scale_layers.append(nn.Linear(input_dim, hidden_dim))
        scale_layers.append(nn.ReLU())
        translate_layers.append(nn.Linear(input_dim, hidden_dim))
        translate_layers.append(nn.ReLU())
        
        # 隐藏层
        for _ in range(num_layers - 2):
            scale_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # 添加dropout防止过拟合
            ])
            translate_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        # 输出层
        scale_layers.append(nn.Linear(hidden_dim, input_dim))
        scale_layers.append(nn.Tanh())  # 限制scale的范围
        translate_layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.scale_net = nn.Sequential(*scale_layers)
        self.translate_net = nn.Sequential(*translate_layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重以提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, reverse=False):
        if not reverse:
            # Forward pass: x -> z
            x_masked = x * self.mask
            scale = self.scale_net(x_masked) * 2.0  # 放大scale范围
            translate = self.translate_net(x_masked)
            
            # 数值稳定的仿射变换
            scale = torch.clamp(scale, -10, 10)  # 防止exp溢出
            z = x_masked + (1 - self.mask) * (x * torch.exp(scale) + translate)
            log_det = torch.sum((1 - self.mask) * scale, dim=-1)
            return z, log_det
        else:
            # Reverse pass: z -> x
            z_masked = x * self.mask
            scale = self.scale_net(z_masked) * 2.0
            translate = self.translate_net(z_masked)
            
            scale = torch.clamp(scale, -10, 10)
            x = z_masked + (1 - self.mask) * (x - translate) * torch.exp(-scale)
            log_det = -torch.sum((1 - self.mask) * scale, dim=-1)
            return x, log_det

class PowerfulNormalizingFlow(nn.Module):
    """
    强大的Normalizing Flow模型，专为高维时间序列设计
    包含降维、Real NVP耦合层和数值稳定性优化
    """
    def __init__(self, input_dim, latent_dim=512, hidden_dim=256, num_coupling_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # === 1. 降维编码器 ===
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # === 2. 升维解码器 ===
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # === 3. Real NVP耦合层 ===
        # 创建交替的mask模式
        masks = []
        for i in range(num_coupling_layers):
            mask = torch.zeros(latent_dim)
            if i % 2 == 0:
                mask[:latent_dim//2] = 1
            else:
                mask[latent_dim//2:] = 1
            masks.append(mask)
        
        self.coupling_layers = nn.ModuleList([
            RealNVPCouplingLayer(latent_dim, hidden_dim//2, masks[i]) 
            for i in range(num_coupling_layers)
        ])
        
        # === 4. 先验分布参数 ===
        self.register_buffer('prior_mean', torch.zeros(latent_dim))
        self.register_buffer('prior_std', torch.ones(latent_dim))
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """编码到潜在空间"""
        # 批量数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用编码器网络
        z_latent = self.encoder(x)
        
        # 输出检查（简化）
        if torch.isnan(z_latent).any() or torch.isinf(z_latent).any():
            z_latent = torch.zeros_like(z_latent)
        
        return z_latent
    
    def decode(self, z_latent):
        """将潜在表示解码回原始空间"""
        # 简化的数值检查
        if torch.isnan(z_latent).any() or torch.isinf(z_latent).any():
            z_latent = torch.zeros_like(z_latent)
        
        # 数值范围检查
        z_latent = torch.clamp(z_latent, -100.0, 100.0)  # 简化范围
        
        x_recon = self.decoder(z_latent)
        
        # 解码结果检查（简化）
        if torch.isnan(x_recon).any() or torch.isinf(x_recon).any():
            x_recon = torch.zeros_like(x_recon)
        
        return x_recon
    
    def forward(self, x):
        """
        前向传播：x -> z_latent -> z_flow
        """
        # 1. 编码到潜在空间
        z_latent = self.encode(x)
        
        # 2. 通过耦合层
        log_det_jacobian_sum = 0
        for coupling_layer in self.coupling_layers:
            z_latent, log_det = coupling_layer(z_latent)
            log_det_jacobian_sum += log_det
        
        return z_latent, log_det_jacobian_sum
    
    def inverse(self, z):
        """
        逆变换：z_flow -> z_latent -> x
        """
        # 1. 逆向通过耦合层
        for coupling_layer in reversed(self.coupling_layers):
            z, _ = coupling_layer(z, reverse=True)
        
        # 2. 解码回原始空间
        x_recon = self.decode(z)
        
        return x_recon
    
    def log_prob(self, x):
        """
        计算对数概率
        """
        try:
            z, log_det_jacobian_sum = self.forward(x)
            
            # 数值稳定性检查
            if torch.isnan(z).any() or torch.isinf(z).any():
                print("警告: Flow变换结果包含NaN/Inf")
                return torch.tensor(-1e6, device=x.device).expand(x.size(0))
            
            if torch.isnan(log_det_jacobian_sum).any() or torch.isinf(log_det_jacobian_sum).any():
                print("警告: 雅可比行列式包含NaN/Inf")
                log_det_jacobian_sum = torch.zeros_like(log_det_jacobian_sum)
            
            # 先验分布的对数概率
            prior = Normal(self.prior_mean, self.prior_std)
            log_prob_prior = prior.log_prob(z).sum(dim=1)
            
            # 检查先验概率
            if torch.isnan(log_prob_prior).any() or torch.isinf(log_prob_prior).any():
                print("警告: 先验概率包含NaN/Inf")
                log_prob_prior = torch.tensor(-1e6, device=x.device).expand(x.size(0))
            
            total_log_prob = log_prob_prior + log_det_jacobian_sum
            
            # 最终检查
            if torch.isnan(total_log_prob).any() or torch.isinf(total_log_prob).any():
                print("警告: 总对数概率包含NaN/Inf，使用备用值")
                total_log_prob = torch.tensor(-1e6, device=x.device).expand(x.size(0))
            
            return total_log_prob
            
        except Exception as e:
            print(f"Flow log_prob计算失败: {e}")
            return torch.tensor(-1e6, device=x.device).expand(x.size(0))
    
    def reconstruct(self, x):
        """
        重构输入：x -> z -> x'
        """
        z, _ = self.forward(x)
        x_recon = self.inverse(z)
        return x_recon
    
    def sample(self, num_samples, device):
        """
        从先验分布采样
        """
        # 从先验分布采样
        z = torch.randn(num_samples, self.latent_dim, device=device)
        z = z * self.prior_std + self.prior_mean
        
        # 逆变换得到样本
        x_samples = self.inverse(z)
        
        return x_samples

class SimpleStableFlow(nn.Module):
    """基于原始项目的简化稳定Flow模型"""
    
    def __init__(self, input_dim, flow_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.flow_layers = nn.ModuleList([
            SimpleFlowLayer(input_dim) for _ in range(flow_layers)
        ])
        self.register_buffer("prior_loc", torch.zeros(input_dim))
        self.register_buffer("prior_scale", torch.ones(input_dim))
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """保守的权重初始化"""
        for m in self.modules():
            if isinstance(m, SimpleFlowLayer):
                # 小初始化避免数值爆炸
                nn.init.normal_(m.scale, mean=0.0, std=0.1)
                nn.init.normal_(m.shift, mean=0.0, std=0.1)
    
    def forward(self, x):
        """前向传播"""
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告: Flow输入包含异常值")
            x = torch.nan_to_num(x, nan=0.0)
        
        log_det_jacobian_sum = 0
        for flow in self.flow_layers:
            x, log_det_jacobian = flow(x)
            log_det_jacobian_sum += log_det_jacobian
        
        return x, log_det_jacobian_sum
    
    def log_prob(self, x):
        """计算对数概率"""
        try:
            z, log_det_jacobian_sum = self.forward(x)
            
            # 数值稳定性检查
            if torch.isnan(z).any() or torch.isinf(z).any():
                return torch.tensor(-1e6, device=x.device).expand(x.size(0))
            
            prior = Normal(self.prior_loc, self.prior_scale)
            log_prob_prior = prior.log_prob(z).sum(dim=1)
            
            total_log_prob = log_prob_prior + log_det_jacobian_sum
            
            # 最终检查
            if torch.isnan(total_log_prob).any() or torch.isinf(total_log_prob).any():
                return torch.tensor(-1e6, device=x.device).expand(x.size(0))
            
            return total_log_prob
            
        except Exception as e:
            print(f"Flow log_prob计算失败: {e}")
            return torch.tensor(-1e6, device=x.device).expand(x.size(0))
    
    def reconstruct(self, x):
        """简单的重构实现"""
        try:
            z, _ = self.forward(x)
            # 逆变换
            for flow in reversed(self.flow_layers):
                scale_clamped = torch.clamp(flow.scale, -2, 2)
                z = (z - flow.shift) / torch.exp(scale_clamped)
            return z
        except Exception as e:
            print(f"Flow重构失败: {e}")
            return x


class SimpleFlowLayer(nn.Module):
    """简化的Flow层，基于原始项目设计"""
    
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(dim) * 0.1)  # 小初始化
        self.shift = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, x):
        """前向传播，限制数值范围"""
        # 严格限制scale范围，避免exp爆炸
        scale = torch.clamp(self.scale, -2, 2)
        z = x * torch.exp(scale) + self.shift
        
        # 雅可比行列式 - 按batch计算
        log_det_jacobian = torch.sum(scale)
        
        # 确保输出形状正确
        if log_det_jacobian.dim() == 0:
            log_det_jacobian = log_det_jacobian.expand(x.size(0))
        
        return z, log_det_jacobian
