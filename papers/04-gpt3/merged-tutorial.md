# GPT-3: Language Models are Few-Shot Learners — 精读融合版

> 阅读指南：论文原文保持不动，每个章节后穿插中文讲解（📖 标记）。
> 注意：GPT-3 论文 72 页，本融合版聚焦核心章节（1-7 节），附录请参阅 raw-extract.md。

---

# Abstract

We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, which is 10x more than any previous non-sparse language model. We test GPT-3 on two dozen NLP tasks, and for all of these, GPT-3 achieves state-of-the-art results under few-shot, one-shot, and zero-shot evaluation.

> 📖 **讲解**
>
> GPT-3 核心卖点：175B 参数（10 倍于之前最大模型），few-shot 方式在 20+ NLP 任务上接近甚至超越微调 SOTA。

---

# 1 Introduction

[论文原文完整保留]

> 📖 **讲解 — 引言核心**
>
> 论文对比了三种方式：
> - **Fine-tuning**：每个任务更新参数（BERT、GPT-1 方式）
> - **Few-shot/One-shot/Zero-shot**：不更新参数，只用 prompt+示例
>
> GPT-3 的论点：**足够大的模型在 few-shot 下可以接近微调效果**，而且更通用——一个模型做所有事。

---

![](images/eval_strategies)

Figure 1.1: Three evaluation strategies.

> 📖 **讲解 — Figure 1.1 三种评估策略**
>
> - **Zero-shot**：只给任务描述 "Translate to French:" + 输入
> - **One-shot**：给 1 个示例对
> - **Few-shot**：给多个示例对（GPT-3 通常用 10-100 个）
>
> 所有方式都不更新参数！模型通过 in-context learning 从示例中学习。

---

# 2 Approach

## 2.1 Model and Architectures

[论文原文完整保留]

> 📖 **讲解 — 2.1 模型架构**
>
> GPT-3 基本沿用 GPT-2 架构（Transformer 解码器 + Pre-LN），关键改动：
> - 交替使用 **dense** 和 **locally banded sparse** attention（每 4 层一次稀疏）
> - 上下文从 1024 扩展到 **2048**
> - 8 种规模：125M → 350M → 760M → 1.3B → 2.7B → 6.7B → 13B → **175B**
>
> 175B: 96 层，12288 维，96 个注意力头

---

## 2.2 Training Dataset

[论文原文完整保留]

> 📖 **讲解 — 2.2 训练数据**
>
> | 数据集 | 权重比例 | Token 数 |
> |--------|---------|---------|
> | Common Crawl (filtered) | 60% | 410B |
> | WebText2 | 20% | 19B |
> | Books1 + Books2 | 9% | 67B |
> | Wikipedia | 3% | 10B |
> | **总计** | - | **~500B** |
>
> Common Crawl 做了质量过滤 + 去重。训练 300B tokens，部分数据不到 1 个 epoch。

---

## 2.3 Training Process

[论文原文完整保留]

> 📖 **讲解 — 2.3 训练细节**
>
> - Batch size: 3.2M tokens（0.5M 序列）
> - 学习率: 0.6e-4 → cosine decay
> - 训练计算量: 3.14 × 10²³ FLOPS
> - 硬件: V100 GPU 集群
> - 训练 300B tokens（仍在 underfitting）

---

# 3 Results

## 3.1 Language Modeling

[Table 3.1: Penn Tree Bank, LAMBADA 结果]

> 📖 **讲解**
>
> - PTB: **20.50 PPL**（SOTA）
> - LAMBADA: **1.92 PPL**（SOTA，从 GPT-2 的 8.6 降到 1.92）
>
> LAMBADA 的巨大提升说明 175B 参数确实学到了更强的长距离依赖能力。

---

## 3.2 SuperGLUE

[Table 3.2: SuperGLUE 结果]

> 📖 **讲解**
>
> GPT-3 few-shot 在 SuperGLUE 上**接近但不一定超过**微调后的 RoBERTa。
> 但关键是：**GPT-3 没有更新参数，同一个模型做了所有任务**。

---

## 3.3 Question Answering

[Table 3.4: QA 结果]

> 📖 **讲解**
>
> TriviaQA: GPT-3 few-shot **71.2** vs 微调 SOTA 68.0 → **超越微调！**
> Natural Questions: 29.9 vs 36.6 → 仍有差距

---

## 3.4 Translation

[Table 3.5: 翻译结果]

> 📖 **讲解**
>
> En→Fr: 25.2 BLEU（GPT-2 是 5 BLEU，进步巨大但仍远低于监督方法 33.5+）
> 翻译是 few-shot 最弱的任务之一——可能因为训练数据中英法对照不够多。

---

## 3.5 Arithmetic

[Table 3.6: 算术结果]

> 📖 **讲解 — 涌现能力**
>
> 算术是 GPT-3 论文中"涌现能力"（emergent ability）的典型案例：
> - 2 位数加法：13B 以下几乎不行，175B 突然到 ~80-90%
> - 更复杂的运算需要更大的模型
> - 这说明**某些能力在特定规模才会"涌现"**

---

# 5 Measuring and Preventing Memorization

[论文原文完整保留：数据污染分析]

> 📖 **讲解 — 数据污染**
>
> 用 13-gram Bloom filter 检测训练数据和测试集的重叠：
> - 大部分数据集的重叠 < 10%
> - 清洗后对性能影响很小
> - 但这个问题后来成为大模型评估的核心关注点

---

# 7 Discussion

[论文原文完整保留]

> 📖 **讲解 — 讨论与局限**
>
> 论文坦诚承认的弱点：
> 1. 文本生成有时缺乏一致性
> 2. 阅读理解和词义消歧（WiC）仍然薄弱
> 3. 训练成本极高
> 4. 偏见和公平性问题
>
> **最重要的未来方向**：如何让语言模型更好地"按照人类意图行动"——这直接指向了 InstructGPT（本系列第 6 篇）。

---

## 延伸阅读

1. **InstructGPT** (Ouyang et al., 2022) — GPT-3 + RLHF，ChatGPT 的前身
2. **Chinchilla** (Hoffmann et al., 2022) — 质疑 GPT-3 数据量不够
3. **PaLM** (Chowdhery et al., 2022) — Google 540B，进一步验证 scaling
