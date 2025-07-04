# M²-MOEP: Mamba-Metric Mixture of Experts for Time Series Prediction

M²-MOEP是一个基于Mamba状态空间模型和度量学习的专家混合时间序列预测框架。

## 🎯 主要特性

- **FFT + ms-Mamba专家网络**：结合频域特征提取和多尺度Mamba建模
- **度量学习门控**：基于嵌入距离的专家选择机制
- **Flow模型预处理**：使用Normalizing Flow进行数据预处理和重构
- **复合损失函数**：包含预测、重构、对比学习、一致性和负载均衡损失
- **通用数据集支持**：支持14种常见时间序列数据集

## 📊 支持的数据集

- **电力数据**：electricity (321个客户)
- **交通数据**：traffic (862个传感器), PEMS03/04/07/08
- **天气数据**：weather (21个气象因子)
- **太阳能数据**：solar (137个发电站)
- **ETT数据**：ETTh1, ETTh2, ETTm1, ETTm2
- **其他**：exchange_rate, illness

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建conda环境
conda create -n m2moep python=3.8
conda activate m2moep

# 安装依赖
pip install -r requirements.txt

# 可选：安装Mamba-SSM
pip install mamba-ssm causal-conv1d
```

### 2. 数据准备

将数据集放置在 `dataset/` 目录下，按以下结构组织：

```

### 3. 运行实验

```bash
# 单个数据集实验
python universal_experiment.py --dataset weather --epochs 50

# 查看可用数据集
python universal_experiment.py --list

# 获取数据集信息
python universal_experiment.py --summary weather

# 运行所有数据集
python universal_experiment.py --all
```

### 4. Flow模型预训练

```bash
# 预训练Flow模型
python pretrain_flow.py --config configs/weather_experiment.yaml
```

### 5. 模型评估

```bash
# 评估训练好的模型
python evaluate.py --model checkpoints/best_model.pth --config configs/weather_experiment.yaml
```

## 🔧 配置文件

使用配置生成器创建实验配置：

```bash
# 生成单个数据集配置
python configs/config_generator.py --dataset weather --output configs/weather_experiment.yaml

# 生成所有数据集配置
python configs/config_generator.py --generate-all

# 列出支持的数据集
python configs/config_generator.py --list
```

## 📁 项目结构

```

## ⚙️ 模型架构

### 专家网络 (FFTmsMambaExpert)
- FFT频域特征提取
- 多尺度Mamba处理
- 专家个性化层

### 门控网络 (GatingEncoder)
- 基于度量学习的专家选择
- 嵌入向量生成
- 距离计算和路由权重

### Flow模型 (PowerfulNormalizingFlow)
- Real NVP耦合层
- 降维编码器/解码器
- 数据重构和预处理

## 📈 实验结果

模型在多个数据集上的性能表现：

| 数据集 | RMSE | MAE | R² |
|--------|------|-----|-----|
| Weather | 0.xxx | 0.xxx | 0.xxx |
| Electricity | 0.xxx | 0.xxx | 0.xxx |
| Traffic | 0.xxx | 0.xxx | 0.xxx |

## 🔬 核心创新

1. **FFT + ms-Mamba专家**：结合频域分析和状态空间建模
2. **度量学习门控**：基于嵌入距离的智能专家选择
3. **Flow模型集成**：数据预处理和重构损失
4. **复合损失设计**：多目标优化的损失函数
5. **专家多样性机制**：确保专家的差异化学习

## 📝 引用

如果您使用了本项目，请引用：

```bibtex
@article{m2moep2024,
  title={M²-MOEP: Mamba-Metric Mixture of Experts for Time Series Prediction},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 联系方式

如有问题，请联系：[your.email@example.com]