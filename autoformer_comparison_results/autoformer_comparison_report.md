# M²-MOEP vs Autoformer 性能比较报告

**实验时间**: 2025-07-05 16:13:42  
**实验描述**: 基于Autoformer论文标准数据集和实验设置的性能比较

## 实验设置

- **输入序列长度**: 96
- **预测长度**: 96, 192, 336, 720
- **数据集**: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Exchange, ILI
- **评估指标**: MSE, MAE, RMSE, R²

## 详细结果

### 按数据集分组的结果


#### Weather

| 预测长度 | MSE | MAE | RMSE | R² | 训练时间 | 参数量 |
|---------|-----|-----|------|----|---------|---------|
| 96 | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED |


## 结论

本实验按照Autoformer论文的标准设置进行，可以直接与Autoformer的发表结果进行比较。

### 关键发现

1. **模型性能**: M²-MOEP在不同数据集和预测长度上的表现
2. **训练效率**: 训练时间和收敛性能
3. **模型复杂度**: 参数量和计算复杂度

### 与Autoformer比较

请将上述结果与Autoformer论文中报告的结果进行比较分析。

---
*本报告由M²-MOEP自动化实验系统生成*
