# GPT-2 图表分析

## Figure 1: Zero-shot 性能随模型规模的变化

核心发现图，展示了 WebText 语言模型在多种 NLP 任务上的 zero-shot 性能如何随参数量增长：
- 横轴：模型参数量（对数坐标，从 ~1M 到 ~1.5B）
- 纵轴：各任务的性能指标
- 任务包括：阅读理解（CoQA）、翻译（WMT-14 Fr-En）、摘要（CNN/DM）、问答（Natural Questions）
- **关键结论**：性能与模型规模呈**对数线性关系**（log-linear），模型越大越好

## Table 2: WebText 语言模型在语言建模基准上的 zero-shot 结果

GPT-2 在 8 个语言建模数据集上的 PPL：
- 7/8 个数据集上取得 SOTA
- 在 WikiText-103 上 PPL 37.5（之前 SOTA ~40）
- 仍然 underfit WebText（训练集 PPL ~18）

## Figure 3: 生成文本示例

展示了 GPT-2 生成的文本样本：
- 提示词 "The Unicorn" → 生成连贯的独角兽故事
- 提示词 "Recycling" → 生成关于回收的连贯段落
- 证明了模型能生成长而连贯的文本

## Table 3: CoQA 数据集结果

Zero-shot 条件下：
- GPT-2 达到 55 F1
- 匹配或超越 4 个 baseline 中的 3 个（它们用了 127K+ 训练样本）

## Figure 4: WMT-14 Fr→En 翻译结果

Zero-shot 翻译性能（通过在法语句子后加英文翻译的方式）：
- GPT-2 达到 5 BLEU
- 远低于监督方法，但证明了语言模型能隐式学习翻译能力
