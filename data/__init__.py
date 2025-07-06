from .data_factory import (
    TimeSeriesDataset,
    data_provider,
    get_dataset_info,
    data_dict
)

# 如果需要使用原始的复杂数据模块，可以按需导入
# from .universal_dataset import UniversalDataModule, UniversalDataset, create_dataset_config

__all__ = [
    'TimeSeriesDataset',
    'data_provider', 
    'get_dataset_info',
    'data_dict'
]
