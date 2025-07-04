import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingEncoder(nn.Module):
    """
    门控编码器：基于度量学习的专家选择
    接受Flow模型的潜在表示作为输入
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 从配置中获取维度信息
        self.latent_dim = config['model']['flow']['latent_dim']  # Flow输出的潜在维度
        self.hidden_dim = config['model']['hidden_dim']
        self.num_experts = config['model']['num_experts']
        self.embedding_dim = config['model']['embedding_dim']
        
        # 门控网络架构
        self.gate_network = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),  # 使用latent_dim作为输入
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_experts)
        )
        
        # 嵌入提取网络
        self.embedding_network = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),  # 使用latent_dim作为输入
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.embedding_dim)
        )
        
        # 专家原型 - 可学习的嵌入向量
        self.expert_prototypes = nn.Parameter(
            torch.randn(self.num_experts, self.embedding_dim)
        )
        nn.init.xavier_uniform_(self.expert_prototypes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z_latent):
        """
        前向传播
        :param z_latent: 潜在表示 [B, latent_dim]
        :return: 专家权重 [B, num_experts]
        """
        batch_size = z_latent.size(0)
        
        # 生成嵌入向量
        embedding = self.embedding_network(z_latent)  # [B, embedding_dim]
        
        # 计算与专家原型的距离
        distances = torch.cdist(
            embedding.unsqueeze(1),  # [B, 1, embedding_dim]
            self.expert_prototypes.unsqueeze(0)  # [1, num_experts, embedding_dim]
        ).squeeze(1)  # [B, num_experts]
        
        # 返回负距离（距离越小，权重越大）
        return -distances
    
    def get_embeddings(self, z_latent):
        """获取嵌入向量（用于度量学习）"""
        return self.embedding_network(z_latent)
