"""
配置生成器 - 修复路径模式版本
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class ConfigGenerator:
    """配置生成器类"""
    
    # 基础配置模板
    BASE_CONFIG = {
        'model': {
            'input_dim': 0,  # 将根据数据集动态设置
            'output_dim': 0,  # 将根据数据集动态设置
            'hidden_dim': 256,
            'num_experts': 4,
            'expert_hidden_dim': 128,
            'embedding_dim': 128,  # 门控网络嵌入维度
            'dropout': 0.1,
            'activation': 'gelu',
            'expert_params': {
                'mamba_d_model': 256,
                'mamba_scales': [1, 2, 4]  # 多尺度处理
            },
            # Flow模型配置
            'flow': {
                'latent_dim': 256,
                'use_pretrained': True,
                'hidden_dim': 256,
                'num_coupling_layers': 6
            },
            # Triplet Loss配置
            'triplet': {
                'margin': 0.5,
                'mining_strategy': 'batch_hard',
                'loss_weight': 1.0,
                'performance_window': 100
            },
            # 多样性配置
            'diversity': {
                'prototype_dim': 64,
                'num_prototypes': 8,  # num_experts * 2
                'diversity_weight': 0.1,
                'force_diversity': True
            },
            # 温度调度配置
            'temperature': {
                'initial': 1.0,
                'min': 0.1,
                'max': 10.0,
                'decay': 0.95,
                'schedule': 'exponential'
            }
        },
        'data': {
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
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'min_lr': 1e-6,
            'patience': 10,
            'flow_model_path': 'flow_model_default.pth',
            'triplet_margin': 0.5,
            'use_reconstruction_loss': True,
            'use_consistency_loss': True,
            'use_load_balancing': True,
            'consistency_temperature': 0.1,
            'aux_loss_weight': 0.01,
            'loss_weights': {
                'init_sigma_rc': 1.0,
                'init_sigma_cl': 1.0,
                'init_sigma_pr': 1.0,
                'init_sigma_consistency': 1.0,
                'init_sigma_balance': 1.0
            }
        },
        'evaluation': {
            'metrics': ['mse', 'mae', 'mape'],
            'save_predictions': True,
            'save_expert_analysis': True
        }
    }
    
    # 数据集特定配置 - 使用正确的路径模式
    DATASET_CONFIGS = {
        'electricity': {
            'description': '321个客户的每小时用电量数据',
            'expected_features': 321,
            'data_path': 'dataset/electricity_321个客户的每小时用电量/electricity.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'traffic': {
            'description': '862个传感器测量的每小时道路占用率',
            'expected_features': 862,
            'data_path': 'dataset/traffic_862个传感器测量的每小时道路占用率/traffic.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'weather': {
            'description': '气象站21个气象因子数据',
            'expected_features': 21,
            'data_path': 'dataset/weather_气象站_21个气象因子/weather.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'ETTh1': {
            'description': '7个因素的变压器温度变化',
            'expected_features': 7,
            'data_path': 'dataset/ETT-small_7个因素的变压器温度变化/ETTh1.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'ETTh2': {
            'description': '7个因素的变压器温度变化',
            'expected_features': 7,
            'data_path': 'dataset/ETT-small_7个因素的变压器温度变化/ETTh2.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'ETTm1': {
            'description': '7个因素的变压器温度变化',
            'expected_features': 7,
            'data_path': 'dataset/ETT-small_7个因素的变压器温度变化/ETTm1.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'ETTm2': {
            'description': '7个因素的变压器温度变化',
            'expected_features': 7,
            'data_path': 'dataset/ETT-small_7个因素的变压器温度变化/ETTm2.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'exchange_rate': {
            'description': '8个国家的汇率变化',
            'expected_features': 8,
            'data_path': 'dataset/exchange_rate_8个国家的汇率变化/exchange_rate.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'illness': {
            'description': '流感患者比例和数量',
            'expected_features': 7,
            'data_path': 'dataset/illness_流感患者比例和数量/national_illness.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'Solar': {
            'description': '137个发电站发电量',
            'expected_features': 137,
            'data_path': 'dataset/Solar_137个发电站发电量/solar_AL.csv',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        },
        'PEMS': {
            'description': '5分钟窗口收集的公共交通网络数据',
            'expected_features': 307,
            'data_path': 'dataset/PEMS_5分钟窗口收集的公共交通网络数据/PEMS04.npz',
            'seq_len': 96,
            'pred_len': 96,
            'scaler_type': 'standard'
        }
    }
    
    @classmethod
    def generate_config(cls, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """
        生成数据集配置
        
        Args:
            dataset_name: 数据集名称
            **kwargs: 额外的配置参数
            
        Returns:
            生成的配置字典
        """
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 获取数据集特定配置
        dataset_config = cls.DATASET_CONFIGS[dataset_name]
        
        # 复制基础配置
        config = cls._deep_copy_dict(cls.BASE_CONFIG)
        
        # 更新模型配置
        config['model']['input_dim'] = dataset_config['expected_features']
        config['model']['output_dim'] = dataset_config['expected_features']
        config['model']['seq_len'] = dataset_config.get('seq_len', 96)
        config['model']['pred_len'] = dataset_config.get('pred_len', 96)
        
        # 更新数据配置
        config['data']['dataset_name'] = dataset_name
        config['data']['data_path'] = dataset_config['data_path']
        config['data']['seq_len'] = dataset_config.get('seq_len', 96)
        config['data']['pred_len'] = dataset_config.get('pred_len', 96)
        config['data']['scaler_type'] = dataset_config.get('scaler_type', 'standard')
        
        # 根据特征数量调整批次大小
        if dataset_config['expected_features'] > 500:
            config['data']['batch_size'] = 16  # 大数据集使用小批次
        elif dataset_config['expected_features'] > 100:
            config['data']['batch_size'] = 32  # 中等数据集
        else:
            config['data']['batch_size'] = 64  # 小数据集使用大批次
        
        # 应用额外参数
        config = cls._apply_kwargs(config, kwargs)
        
        return config
    
    @classmethod
    def _deep_copy_dict(cls, d: Dict) -> Dict:
        """深拷贝字典"""
        import copy
        return copy.deepcopy(d)
    
    @classmethod
    def _apply_kwargs(cls, config: Dict, kwargs: Dict) -> Dict:
        """应用额外的配置参数"""
        for key, value in kwargs.items():
            if '.' in key:
                # 支持嵌套键，如 'model.hidden_dim'
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        return config
    
    @classmethod
    def save_config(cls, config: Dict, filepath: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def load_config(cls, filepath: str) -> Dict:
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def get_supported_datasets(cls) -> List[str]:
        """获取支持的数据集列表"""
        return list(cls.DATASET_CONFIGS.keys())
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        return cls.DATASET_CONFIGS[dataset_name].copy()


def main():
    """主函数：生成所有数据集的配置文件"""
    generator = ConfigGenerator()
    
    # 创建配置目录
    os.makedirs('configs/datasets', exist_ok=True)
    
    # 为每个数据集生成配置
    for dataset_name in generator.get_supported_datasets():
        print(f"生成数据集 {dataset_name} 的配置...")
        
        # 生成配置
        config = generator.generate_config(dataset_name)
        
        # 保存配置
        config_path = f'configs/datasets/{dataset_name}_config.yaml'
        generator.save_config(config, config_path)
        
        print(f"配置已保存到: {config_path}")
    
    print("所有数据集配置生成完成！")


if __name__ == '__main__':
    main()