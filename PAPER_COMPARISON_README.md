# M²-MOEP 论文对比实验指南

## 概述

本指南帮助您运行 M²-MOEP 模型与论文 "Non-autoregressive Conditional Diffusion Models for Time Series Prediction" 的对比实验。

## 重合数据集

通过分析，您的项目与目标论文有以下 **6个重合数据集**：

| 论文数据集 | 项目数据集 | 描述 |
|-----------|-----------|------|
| Weather | `weather` | 气象站21个气象因子数据 |
| ETTm1 | `ETTm1` | 7个因素的变压器温度变化 |
| Traffic | `traffic` | 862个传感器道路占用率数据 |
| Electricity | `electricity` | 321个客户用电量数据 |
| ETTh1 | `ETTh1` | 7个因素的变压器温度变化 |
| Exchange | `exchange_rate` | 8个国家汇率变化数据 |

## 论文基准结果

### 单变量设置 (MSE)
| 数据集 | TimeDiff | DLinear | FiLM |
|-------|----------|---------|------|
| Weather | 0.002 | 0.168 | 0.007 |
| ETTm1 | 0.040 | 0.041 | 0.038 |
| Traffic | 0.121 | 0.139 | 0.198 |
| Electricity | 0.232 | 0.244 | 0.260 |
| ETTh1 | 0.066 | 0.078 | 0.070 |
| Exchange | 0.017 | 0.017 | 0.018 |

### 多变量设置 (MSE)
| 数据集 | TimeDiff | DLinear | FiLM |
|-------|----------|---------|------|
| Weather | 0.311 | 0.488 | 0.327 |
| ETTm1 | 0.336 | 0.345 | 0.347 |
| Traffic | 0.564 | 0.389 | 0.628 |
| Electricity | 0.193 | 0.215 | 0.210 |
| ETTh1 | 0.407 | 0.445 | 0.426 |
| Exchange | 0.018 | 0.022 | 0.016 |

## 使用步骤

### 1. 检查数据集可用性

首先运行数据集检查脚本：

```bash
cd zhouzp/m-mamba/m2moep
python check_datasets.py
```

这将显示：
- 哪些数据集可用
- 每个数据集的基本信息（特征数、样本数等）
- 运行建议

### 2. 运行对比实验

#### 选项A：运行所有对比实验（推荐）
```bash
python paper_comparison_experiment.py
```

#### 选项B：快速测试（少量epochs）
```bash
python paper_comparison_experiment.py --epochs 10
```

#### 选项C：只运行多变量实验
```bash
python paper_comparison_experiment.py --modes multivariate
```

#### 选项D：只运行单变量实验
```bash
python paper_comparison_experiment.py --modes univariate
```

#### 选项E：运行指定数据集
```bash
python paper_comparison_experiment.py --datasets weather traffic electricity
```

### 3. 查看结果

实验完成后，将生成以下文件：

1. **`paper_comparison_results_YYYYMMDD_HHMMSS.json`** - 详细的实验结果
2. **`paper_comparison_report_YYYYMMDD_HHMMSS.txt`** - 对比报告
3. **`comparison_table_multivariate.tex`** - LaTeX格式的对比表格
4. **`paper_comparison_YYYYMMDD_HHMMSS.log`** - 实验日志

### 4. 结果分析

对比报告包含：
- 每个数据集的MSE对比
- 相对于基准方法的改进百分比
- 统计信息汇总

## 实验配置

### 默认配置
- **序列长度**: 96
- **预测长度**: 96
- **批次大小**: 32 (多变量) / 16 (单变量)
- **训练轮数**: 50
- **优化器**: Adam
- **学习率**: 0.0001

### 自定义配置
您可以通过命令行参数调整：
```bash
python paper_comparison_experiment.py \
    --epochs 100 \
    --modes multivariate \
    --datasets weather traffic
```

## 预期结果

基于您的 M²-MOEP 模型架构，预期在以下方面有所改进：

1. **多尺度时间建模**: Mamba架构的多尺度处理能力
2. **专家混合**: 不同专家处理不同时间模式
3. **Flow重构**: 标准化流提供更好的数据表示
4. **自适应门控**: 动态选择最适合的专家

## 故障排除

### 常见问题

1. **数据集不可用**
   - 检查 `dataset/` 目录下是否有对应的数据文件
   - 确保数据文件格式正确 (CSV)

2. **内存不足**
   - 减少批次大小：`--batch-size 8`
   - 使用更少的epochs：`--epochs 10`

3. **Flow模型未找到**
   - 脚本会自动预训练Flow模型
   - 确保有足够的磁盘空间

### 调试模式

启用详细日志：
```bash
python paper_comparison_experiment.py --epochs 5 --datasets weather
```

## 学术使用建议

### 论文撰写
1. 使用生成的LaTeX表格直接插入论文
2. 引用对比报告中的改进百分比
3. 分析不同数据集上的性能差异

### 进一步分析
1. 专家选择模式分析
2. 不同时间尺度上的性能
3. 计算复杂度对比
4. 收敛性分析

## 技术细节

### 模型架构
- **Mamba骨干网络**: 多尺度时间序列建模
- **专家混合 (MoE)**: 4个专家，top-k=3选择
- **标准化流**: 6层耦合层，256维潜在空间
- **三元组损失**: 批次硬挖掘策略

### 评估指标
- **MSE**: 均方误差（主要对比指标）
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数

## 联系与支持

如果遇到问题：
1. 检查日志文件中的错误信息
2. 确认数据集格式和路径
3. 验证模型依赖项是否正确安装

祝您实验顺利！🚀 