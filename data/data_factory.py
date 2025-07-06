"""
简化的数据工厂模块
参考iTransformer和Autoformer的设计理念
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Tuple
import torch


class TimeSeriesDataset(Dataset):
    """统一的时间序列数据集类"""
    
    def __init__(self, root_path: str, data_path: str, flag: str = 'train', 
                 size: List[int] = [96, 48, 24], features: str = 'M', 
                 target: str = 'OT', scale: bool = True, inverse: bool = False):
        """
        Args:
            root_path: 数据根目录
            data_path: 数据文件名
            flag: 数据集类型 ('train', 'val', 'test')
            size: [seq_len, label_len, pred_len]
            features: 特征类型 ('M': 多元->多元, 'S': 单元->单元, 'MS': 多元->单元)
            target: 目标列名
            scale: 是否标准化
            inverse: 是否反标准化
        """
        self.seq_len, self.label_len, self.pred_len = size
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag  # 正确设置flag属性
        
        self.scaler = StandardScaler()
        self.__read_data__()
        
    def __read_data__(self):
        """读取数据"""
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 移除时间列（如果存在）
        if 'date' in df_raw.columns:
            df_raw = df_raw.drop(['date'], axis=1)
        
        # 选择特征
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns.tolist()
            if self.target in cols_data:
                cols_data.remove(self.target)
            df_data = df_raw[cols_data + [self.target]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        # 数据分割
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_val = len(df_data) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_data)]
        
        train_data = df_data[border1s[0]:border2s[0]]
        val_data = df_data[border1s[1]:border2s[1]]
        test_data = df_data[border1s[2]:border2s[2]]
        
        # 标准化
        if self.scale:
            self.scaler.fit(train_data.values)
            train_data = self.scaler.transform(train_data.values)
            val_data = self.scaler.transform(val_data.values)
            test_data = self.scaler.transform(test_data.values)
        else:
            train_data = train_data.values
            val_data = val_data.values  
            test_data = test_data.values
        
        # 确保数据类型为float32
        train_data = train_data.astype(np.float32)
        val_data = val_data.astype(np.float32)
        test_data = test_data.astype(np.float32)
        
        # 选择数据集
        if self.flag == 'train':
            self.data_x = train_data
            self.data_y = train_data
        elif self.flag == 'val':
            self.data_x = val_data
            self.data_y = val_data
        elif self.flag == 'test':
            self.data_x = test_data
            self.data_y = test_data
        
        # 记录数据信息
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x)
        
    def __getitem__(self, index):
        """获取数据项"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # 确保数据类型为float32
        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.astype(np.float32)
        
        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        """反标准化"""
        return self.scaler.inverse_transform(data)


# 数据集字典 - 类似iTransformer的设计
data_dict = {
    'electricity': TimeSeriesDataset,
    'weather': TimeSeriesDataset,
    'traffic': TimeSeriesDataset,
    'ETTh1': TimeSeriesDataset,
    'ETTh2': TimeSeriesDataset,
    'ETTm1': TimeSeriesDataset,
    'ETTm2': TimeSeriesDataset,
    'custom': TimeSeriesDataset,
}


def data_provider(args, flag):
    """
    数据提供者函数 - 参考iTransformer设计
    
    Args:
        args: 参数对象
        flag: 数据集类型 ('train', 'val', 'test')
    
    Returns:
        data_set: 数据集对象
        data_loader: 数据加载器
    """
    Data = data_dict.get(args.data, TimeSeriesDataset)
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=getattr(args, 'scale', True),
        inverse=getattr(args, 'inverse', False)
    )
    
    print(f"{flag} dataset length: {len(data_set)}")
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, 'num_workers', 2),
        drop_last=drop_last
    )
    
    return data_set, data_loader


def get_dataset_info(args):
    """获取数据集信息"""
    # 临时创建数据集以获取信息
    temp_dataset = data_dict.get(args.data, TimeSeriesDataset)(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target
    )
    
    return {
        'enc_in': temp_dataset.enc_in,
        'num_features': temp_dataset.enc_in,
        'num_samples': temp_dataset.tot_len,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len
    } 