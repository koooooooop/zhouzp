"""
通用数据集模块 - 完整复原版本
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Optional, Tuple

from models.flow import PowerfulNormalizingFlow


class UniversalDataset(Dataset):
    """通用数据集类"""
    
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, mode: str = 'train'):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        
        # 计算可用的序列数量
        self.num_sequences = len(data) - seq_len - pred_len + 1
        
        if self.num_sequences <= 0:
            raise ValueError(f"数据长度不足以生成序列: 数据长度={len(data)}, 需要={seq_len + pred_len}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # 输入序列
        seq_x = self.data[idx:idx + self.seq_len]
        # 目标序列
        seq_y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)


class UniversalDataModule:
    """通用数据模块"""
    
    def __init__(self, config: Dict, for_pretraining: bool = False):
        self.config = config
        self.for_pretraining = for_pretraining
        
        # 数据配置
        data_config = config['data']
        self.data_path = data_config['data_path']
        self.seq_len = data_config['seq_len']
        self.pred_len = data_config['pred_len']
        self.batch_size = data_config['batch_size']
        self.num_workers = data_config.get('num_workers', 2)
        self.pin_memory = data_config.get('pin_memory', True)
        self.train_ratio = data_config.get('train_ratio', 0.7)
        self.val_ratio = data_config.get('val_ratio', 0.2)
        self.test_ratio = data_config.get('test_ratio', 0.1)
        self.scaler_type = data_config.get('scaler_type', 'standard')
        self.normalize = data_config.get('normalize', True)
        
        # 数据存储
        self.raw_data = None
        self.normalized_data = None
        self.scaler = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.data_info = {}
        self.flow_model = None
        
        # 加载数据
        self._load_data()
        
        # 加载Flow模型（如果不是预训练模式）
        if not for_pretraining:
            self._load_flow_model()
    
    def _find_data_files(self) -> List[str]:
        """查找数据文件"""
        print(f"正在查找数据文件: {self.data_path}")
        
        # 检查是否是合成数据
        if self.data_path == 'synthetic' or self.data_path is None:
            print("使用合成数据")
            return ['synthetic']
        
        if '*' in self.data_path:
            # 使用glob模式匹配
            files = glob.glob(self.data_path)
        else:
            # 检查是否是目录
            if os.path.isdir(self.data_path):
                # 查找目录中的所有支持的数据文件
                files = []
                for ext in ['*.csv', '*.txt', '*.npz']:
                    files.extend(glob.glob(os.path.join(self.data_path, ext)))
            else:
                # 直接路径
                files = [self.data_path] if os.path.exists(self.data_path) else []
        
        if not files:
            raise FileNotFoundError(f"未找到数据文件: {self.data_path}")
        
        print(f"找到数据文件: {files}")
        return sorted(files)
    
    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """加载单个数据文件"""
        try:
            print(f"正在加载文件: {file_path}")
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.txt'):
                # 处理txt文件，假设是空格或制表符分隔
                df = pd.read_csv(file_path, sep='\s+', header=None)
            elif file_path.endswith('.npz'):
                # 处理npz文件
                data = np.load(file_path)
                # 假设数据在'data'键中，或者取第一个数组
                if 'data' in data:
                    array_data = data['data']
                else:
                    # 取第一个数组
                    array_data = data[list(data.keys())[0]]
                
                # 如果是3D数组，展平为2D
                if array_data.ndim == 3:
                    # 假设形状为 (time, nodes, features)
                    array_data = array_data.reshape(array_data.shape[0], -1)
                
                df = pd.DataFrame(array_data)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            print(f"文件形状: {df.shape}")
            print(f"列数: {df.shape[1]}")
            
            # 移除非数值列
            if file_path.endswith('.csv') or file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_columns]
                print(f"数值列数量: {len(numeric_columns)}")
            
            # 处理缺失值
            if df.isnull().sum().sum() > 0:
                print("检测到缺失值，正在处理...")
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"加载文件失败 {file_path}: {str(e)}")
    
    def _load_data(self):
        """加载和预处理数据"""
        try:
            # 查找数据文件
            data_files = self._find_data_files()
            
            # 加载所有文件
            dataframes = []
            for file_path in data_files:
                if file_path == 'synthetic':
                    # 生成合成数据
                    df = self._generate_synthetic_data()
                else:
                    df = self._load_single_file(file_path)
                dataframes.append(df)
            
            # 合并数据
            if len(dataframes) == 1:
                combined_df = dataframes[0]
            else:
                # 按时间轴合并
                combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
            
            # 转换为numpy数组
            self.raw_data = combined_df.values.astype(np.float32)
            
            print(f"合并后数据形状: {self.raw_data.shape}")
            
            # 数据信息
            self.data_info = {
                'num_features': self.raw_data.shape[1],
                'num_samples': self.raw_data.shape[0],
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'data_files': data_files,
                'data_path': self.data_path
            }
            
            # 数据标准化
            if self.normalize:
                self._normalize_data()
            else:
                self.normalized_data = self.raw_data.copy()
            
            # 数据分割
            self._split_data()
            
            print("数据加载完成")
            
        except Exception as e:
            raise RuntimeError(f"数据加载失败: {str(e)}")
    
    def _normalize_data(self):
        """数据标准化"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler(quantile_range=(25, 75))
        else:
            raise ValueError(f"不支持的标准化类型: {self.scaler_type}")
        
        # 拟合并转换数据
        self.normalized_data = self.scaler.fit_transform(self.raw_data)
        
        # 更新数据信息
        self.data_info['scaler_type'] = self.scaler_type
        self.data_info['scaler'] = self.scaler
        
        print(f"数据标准化完成，使用 {self.scaler_type} 标准化")
    
    def _split_data(self):
        """数据分割"""
        total_len = len(self.normalized_data)
        
        # 计算分割点
        train_end = int(total_len * self.train_ratio)
        val_end = int(total_len * (self.train_ratio + self.val_ratio))
        
        # 分割数据
        self.train_data = self.normalized_data[:train_end]
        self.val_data = self.normalized_data[train_end:val_end]
        self.test_data = self.normalized_data[val_end:]
        
        # 更新数据信息
        self.data_info.update({
            'train_size': len(self.train_data),
            'val_size': len(self.val_data),
            'test_size': len(self.test_data)
        })
        
        print(f"数据分割完成 - 训练集: {len(self.train_data)}, 验证集: {len(self.val_data)}, 测试集: {len(self.test_data)}")
    
    def _load_flow_model(self):
        """加载Flow模型"""
        flow_model_path = self.config['training'].get('flow_model_path', 'flow_model_default.pth')
        
        if os.path.exists(flow_model_path):
            try:
                from models.flow import PowerfulNormalizingFlow
                
                # 计算正确的输入维度
                input_dim = self.data_info['num_features'] * self.seq_len
                latent_dim = min(512, input_dim // 4)  # 自适应潜在维度
                
                # 创建Flow模型（使用正确的参数）
                self.flow_model = PowerfulNormalizingFlow(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    hidden_dim=256,
                    num_coupling_layers=6
                )
                
                # 加载权重
                checkpoint = torch.load(flow_model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.flow_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.flow_model.load_state_dict(checkpoint)
                
                self.flow_model.eval()
                print(f"Flow模型加载成功: {flow_model_path}")
                print(f"  输入维度: {input_dim}, 潜在维度: {latent_dim}")
                
            except Exception as e:
                print(f"Flow模型加载失败: {e}")
                self.flow_model = None
        else:
            print(f"Flow模型文件不存在: {flow_model_path}")
            self.flow_model = None
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        return self.data_info.copy()
    
    def get_train_dataset(self) -> UniversalDataset:
        """获取训练数据集"""
        return UniversalDataset(self.train_data, self.seq_len, self.pred_len, 'train')
    
    def get_val_dataset(self) -> UniversalDataset:
        """获取验证数据集"""
        return UniversalDataset(self.val_data, self.seq_len, self.pred_len, 'val')
    
    def get_test_dataset(self) -> UniversalDataset:
        """获取测试数据集"""
        return UniversalDataset(self.test_data, self.seq_len, self.pred_len, 'test')
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        dataset = self.get_train_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        dataset = self.get_val_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器"""
        dataset = self.get_test_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反标准化"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data
    
    def get_flow_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """获取Flow模型重构"""
        if self.flow_model is None:
            return torch.empty(0)  # 返回空张量而不是零张量
        
        try:
            with torch.no_grad():
                original_shape = x.shape  # [B, seq_len, input_dim]
                batch_size = original_shape[0]
                
                # 展平输入用于Flow模型
                x_flat = x.view(batch_size, -1)  # [B, seq_len * input_dim]
                
                # Flow模型重构
                x_recon_flat = self.flow_model.reconstruct(x_flat)  # [B, seq_len * input_dim]
                
                # 恢复原始形状
                x_reconstructed = x_recon_flat.view(original_shape)  # [B, seq_len, input_dim]
                
                return x_reconstructed
                
        except Exception as e:
            print(f"Flow重构失败: {e}")
            return torch.empty(0)  # 返回空张量表示失败

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """生成合成数据"""
        print("正在生成合成数据...")
        
        # 获取合成数据配置
        data_config = self.config['data']
        num_samples = data_config.get('synthetic_samples', 1000)
        num_features = self.config['model']['input_dim']
        noise_level = data_config.get('noise_level', 0.1)
        
        # 生成时间序列数据
        t = np.linspace(0, 4 * np.pi, num_samples)
        
        # 创建多个特征
        data = np.zeros((num_samples, num_features))
        
        for i in range(num_features):
            # 不同的信号模式
            if i % 4 == 0:
                # 正弦波
                signal = np.sin(t + i * 0.5)
            elif i % 4 == 1:
                # 余弦波
                signal = np.cos(t + i * 0.3)
            elif i % 4 == 2:
                # 锯齿波
                signal = 2 * (t % (2 * np.pi)) / (2 * np.pi) - 1
            else:
                # 随机游走
                signal = np.cumsum(np.random.randn(num_samples) * 0.1)
            
            # 添加噪声
            noise = np.random.randn(num_samples) * noise_level
            data[:, i] = signal + noise
        
        print(f"生成合成数据形状: {data.shape}")
        return pd.DataFrame(data)


def create_dataset_config(dataset_name: str, data_path: str, **kwargs) -> Dict:
    """创建数据集配置的便捷函数"""
    from configs.config_generator import ConfigGenerator
    
    config = ConfigGenerator.generate_config(dataset_name, **kwargs)
    config['data']['data_path'] = data_path
    
    return config


def test_dataset_loading():
    """测试数据集加载"""
    # 测试配置
    test_config = {
        'data': {
            'data_path': 'dataset/electricity_*',
            'seq_len': 96,
            'pred_len': 96,
            'batch_size': 32,
            'num_workers': 2,
            'pin_memory': True,
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1,
            'scaler_type': 'standard',
            'normalize': True
        },
        'training': {
            'flow_model_path': 'flow_model_default.pth'
        }
    }
    
    try:
        # 创建数据模块
        data_module = UniversalDataModule(test_config)
        
        # 获取数据信息
        info = data_module.get_dataset_info()
        print("数据集信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 测试数据加载器
        train_loader = data_module.get_train_loader()
        print(f"\n训练集批次数: {len(train_loader)}")
        
        # 测试一个批次
        for batch_x, batch_y in train_loader:
            print(f"输入形状: {batch_x.shape}")
            print(f"目标形状: {batch_y.shape}")
            break
            
        print("数据集加载测试成功！")
        
    except Exception as e:
        print(f"数据集加载测试失败: {e}")


if __name__ == '__main__':
    test_dataset_loading()