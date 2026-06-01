# LoRA: Low-Rank Adaptation of Large Language Models — 精读融合版

> 阅读指南：论文原文保持不动，穿插中文讲解（📖 标记）。

---

# Abstract

We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture [...] LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.

> 📖 **讲解 · 鸟瞰**
>
> **一句话**：冻结预训练权重 + 低秩分解 = 0.01% 参数达到全量微调效果。
>
> **核心优势**：参数少（0.01%）、训练快（3x）、推理零延迟。

---

# 1. Introduction

[论文原文...]

> 📖 **讲解 · 精读 · 问题**
>
> 全量微调 GPT-3 175B：每个任务需要 175B 参数的副本。10 个任务 = 1.75T 参数。
>
> LoRA：每个任务只需 ~18M 参数（0.01%），10 个任务 = ~180M + 共享 175B 基础模型。

![](images/c18d09578dd3be6848a7f5e3dcbf498811633fddd8baefd539a94da7966796bd.jpg)

Figure 1: Reparametrization.

> 📖 **讲解 · 图表精读（Figure 1——LoRA 核心图）**
>
> $h = W_0 x + BAx$
> - $W_0$：冻结（蓝色，不更新）
> - $B$：$d \times r$，零初始化
> - $A$：$r \times k$，高斯初始化
> - 部署时合并：$W_{\text{deploy}} = W_0 + BA$

---

# 4. Our Method

## 4.1 Low-Rank-Parametrized Update Matrices

[论文原文...]

> 📖 **讲解 · 精读 · 核心公式**
>
> $$h = W_0 x + BAx$$
>
> - $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$
> - $B$ 零初始化 → 训练开始 $\Delta W = 0$ → 不破坏预训练
> - 参数量：$(d+k) \times r$ vs 全量的 $d \times k$
>
> **缩放因子**：$\Delta W = \frac{\alpha}{r} BA$（固定 $\alpha$，变 $r$ 不用调学习率）
>
> **应用位置**：对 $W_q$ 和 $W_v$ 效果最好
>
> ❓ **为什么 Q 和 V 最重要？** Q 控制"关注什么"，V 控制"传递什么"——微调主要改这两个。

---

# 5. Experiments

[论文原文...]

> 📖 **讲解 · 精读 · 核心结果**
>
> **GPT-3 175B**：LoRA r=4 只需 ~18M 参数，匹配全量微调
>
> **r 的选择**：r=1-2 在某些任务就够用！说明微调权重变化确实是低秩的
>
> **推理延迟**：LoRA = 全量微调（合并后一样），Adapter +2-30%

---

> 📖 **讲解 · 知识网络**
>
> **LoRA 的遗产**：开源微调标准（Stable Diffusion、LLaMA 都用 LoRA）
>
> **后续发展**：QLoRA（4-bit 量化）、LoRA+（不对称初始化）、DoRA（幅度+方向分解）
>
> **面试核心**：
> - Q: LoRA 怎么做到零延迟？A: $W_0 + BA$ 预先合并
> - Q: 为什么 B 零初始化？A: 训练开始 $\Delta W = 0$，不破坏预训练
