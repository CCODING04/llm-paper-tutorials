# Attention Is All You Need — 图表分析

## Figure 1: Transformer 整体架构
**文件**: d018247de7540bbbd7d638e7b3a9aa21d04567cb8492ac4ce39dc5526098c0b6.jpg

**详细描述**:
这张图展示了 Transformer 模型的完整架构，分为左右两个主要部分：**左侧是编码器（Encoder）**，**右侧是解码器（Decoder）**。

编码器（N×6层）：Input Embedding + Positional Encoding → Multi-Head Attention (Add&Norm) → Feed Forward (Add&Norm)
解码器（N×6层）：Output Embedding + Positional Encoding → Masked Multi-Head Attention (Add&Norm) → Multi-Head Attention over encoder output (Add&Norm) → Feed Forward (Add&Norm) → Linear → Softmax

**教学要点**：残差连接保证梯度流动，层归一化稳定训练，掩码保证自回归。

## Figure 2a: 缩放点积注意力
**文件**: 8e2f2d53c630b12bf40ffec663fc295c3e6e4d64b0fd9109a21e79c97ebecbba.jpg

Q·K^T → Scale(÷√d_k) → Mask(opt.) → Softmax → ×V

**教学要点**：缩放防止梯度消失，掩码防止信息泄露。

## Figure 2b: 多头注意力
**文件**: be7b5c0abeb59a06bbff05163bc1662b675ddb4c993454cc3389bd4cba6d9d6f.jpg

V/K/Q 各经 Linear 投影 → h 个头并行 Scaled Dot-Product Attention → Concat → Linear 输出

**教学要点**：不同头学习不同信息类型，总计算量与单头相当。

## Figure 3: 长距离依赖可视化
**文件**: b94091729c229b14a4a2f46f367189e0b69a5fa8daf3cc682be4bef0d269a0ab.jpg

编码器第5层对"making"的注意力：连接到"more""difficult""laws""process"等，展示跨距离依赖捕捉。

## Figure 4: 代词消解
**文件**: 5f0519f587dfa1718c4bb9aeb8f44e1307c30bd07d803bf6ef7aee5b5b386a5a.jpg

Head 5/6 对"its"的注意力指向"Law"和"application"，模型自动学会代词消解。

## Figure 5: 句法结构
**文件**: 14f54e462f2619a932754f93093ce259fc9e297689d5cfbbf0e77e01c3984c97.jpg

两个不同头学到互补的句法信息：主谓关系 vs 修饰关系。
