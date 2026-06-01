# GPT-3 图表分析

## Figure 1.1: 三种评估策略对比
- **Zero-shot**：仅给任务描述（"Translate to French:"）
- **One-shot**：给一个示例
- **Few-shot**：给多个示例
- 论文核心：GPT-3 通过 few-shot 在不用任何梯度更新的情况下完成任务

## Figure 1.2: GPT-3 模型规模
- 8 种不同规模的模型
- 从 125M 到 175B 参数
- 上下文长度统一为 2048
- 关键：所有模型用相同的超参数框架训练

## Figure 2.1: Scale Plots（核心结果图）
- 横轴：模型参数量（对数坐标）
- 纵轴：任务性能
- 几乎所有任务都呈现 **smooth power-law scaling**
- Few-shot 性能远优于 zero-shot
- 175B 模型在多数任务上接近或超过 SOTA

## Figure 3.1: Training Curves
- 训练 loss 随计算量的变化
- 更大模型收敛更快（sample efficiency 更高）
- 175B 模型训练到 300B tokens 仍未饱和

## Figure 3.2: Training Compute
- 各模型的训练计算量对比
- GPT-3 175B: 3.14E23 FLOPS
- 约 355 GPU-years（V100）

## Figure 8.1: Data Contamination
- 训练数据和测试集的重叠分析
- Bloom filter 检测 13-gram 重叠
- 清洗后性能影响很小
