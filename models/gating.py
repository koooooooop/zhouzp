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
        
        # 统一从 model.flow.latent_dim 获取维度信息
        flow_config = config['model'].get('flow', {})
        self.latent_dim = flow_config.get('latent_dim', 256)  # 统一数据源
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
    
    def forward(self, z_latent, use_gate_network=False):
        """
        前向传播 - 严格按照文档的可学习专家原型设计
        """
        batch_size = z_latent.size(0)
        
        # 按文档设计：默认使用基于原型距离的门控
        if use_gate_network:
            # 传统MLP门控策略（保留作为备选）
            return self.gate_network(z_latent)
        else:
            # 基于原型距离的门控策略（文档核心设计）
            embedding = self.embedding_network(z_latent)
            
            # 计算与专家原型的距离（文档Algorithm 1 Line 8-9）
            distances = torch.cdist(
                embedding.unsqueeze(1),
                self.expert_prototypes.unsqueeze(0)
            ).squeeze(1)
            
            # 返回负距离（距离越小，权重越大）
            return -distances
    
    def get_embeddings(self, z_latent):
        """获取嵌入向量（用于度量学习）"""
        return self.embedding_network(z_latent)
