# LoRA — 精读融合版

> 论文原文保持不动，穿插中文讲解（📖 标记）。

---

# Abstract

We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture [...] LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.

> 📖 **讲解 · 鸟瞰**
>
> **一句话**：冻结 W₀ + 低秩分解 BA = 0.01% 参数匹配全量微调，推理零延迟。
>
> **核心数字**：参数减少 10,000x，显存减少 3x，推理零延迟。

---

# 1. Introduction

[原文...]

> 📖 **讲解 · 精读 · 问题定义**
>
> 全量微调 GPT-3 175B：每个任务 175B 参数副本。10 任务 = 1.75T。
>
> 现有方法的问题：Adapter 有延迟，Prefix Tuning 占序列长度。
>
> LoRA：每个任务只需 ~18M（0.01%）+ 共享 175B 基础模型。

![](images/c18d09578dd3be6848a7f5e3dcbf498811633fddd8baefd539a94da7966796bd.jpg)

Figure 1: Reparametrization

> 📖 **讲解 · 图表精读（Figure 1——LoRA 核心图）**
>
> $h = W_0 x + BAx$
> - $W_0$：冻结（蓝色）
> - $B$：$d \times r$，**零初始化**
> - $A$：$r \times k$，高斯初始化
> - 部署合并：$W_{\text{deploy}} = W_0 + BA$（一次性，推理零延迟）
>
> ❓ **为什么 B 零初始化？** 训练开始 ΔW = 0 → 不破坏预训练知识。

---

# 3. Aren't Existing Solutions Good Enough?

[原文：Adapter latency + Prefix Tuning issues...]

Table 1: 推理延迟

> 📖 **讲解 · 图表精读（Table 1——推理延迟对比）**
>
> 在线推理（B=1, S=128）时 Adapter 延迟增加 **30%**。
> LoRA = 全量微调（合并后完全一样）。
>
> ❓ **为什么大 batch 时 Adapter 延迟小？** GPU 并行度高，额外层被摊薄。

---

# 4. Our Method

## 4.1 Low-Rank-Parametrized Update Matrices

[原文：$h = W_0 x + BAx$...]

> 📖 **讲解 · 精读 · 核心公式详解**
>
> $$h = W_0 x + \frac{\alpha}{r} BAx$$
>
> **参数量**：$(d+k) \times r$ vs 全量的 $d \times k$ → 压缩 ~768x（GPT-3 $W_q$）
>
> **缩放因子 α/r**：固定 α，变 r 不用调学习率。实践中 α=16 或 α=2r。
>
> **应用位置**：论文发现 $W_q + W_v$ 效果最好——Q 控制"关注什么"，V 控制"传递什么"。

## 4.2 Applying LoRA to Transformer

[原文...]

> 📖 **讲解 · 精读 · 实践要点**
>
> - 只对注意力层加 LoRA（不对 MLP）→ 效果好且参数少
> - 实践中常用 r=8, α=16
> - GPT-3 175B: 96 层 × 2 (Q+V) × 196K = ~37.7M 可训练参数

---

# 5. Empirical Experiments

## 5.1 RoBERTa / DeBERTa / GPT-2

[原文...]

> 📖 **讲解 · 精读 · 小模型验证**
>
> LoRA 在 RoBERTa、DeBERTa、GPT-2 上都匹配或超过全量微调。
> 初步验证"低秩假设"成立。

## 5.2 GPT-3 175B

[原文...]

> 📖 **讲解 · 精读 · 旗舰结果**
>
> LoRA r=4, Q+V: WikiSQL 73.8%, MNLI 89.2% → 匹配全量微调
> Adapter: 72.2%, 88.3% → 落后
> Prefix Tuning: 69.7%, 87.5% → 更落后
>
> **用 0.01% 参数达到全量微调效果！**

---

# 6. Understanding the Low-Rank Updates

## 6.1 Which Weight Matrices?

Table 2: Q+V 效果最好

> 📖 **讲解 · 消融实验**
>
> Q+V > Q+K > Q only > K only > V only > O only
>
> **为什么 Q 和 V 最重要？** Q="关注什么"，V="传递什么"——微调主要改这两个。

## 6.2 Rank Selection

Table 3: r=1 就够用

> 📖 **讲解 · Rank 分析**
>
> r=1 WikiSQL 73.5% vs r=64 73.7% → 差距只有 0.2%
>
> **微调的 ΔW 确实是极低秩的。**

## 6.3 Subspace Similarity

Figure 4/5: 子空间分析

> 📖 **讲解 · 子空间分析**
>
> r=8 的 top 方向 ⊂ r=64 的子空间 → 存在"最优低秩子空间"
>
> ΔW 的方向和 W₀ **正交** → LoRA 学到的是新方向，不是放大已有方向

---

> 📖 **讲解 · 知识网络**
>
> **LoRA 的遗产**：开源微调标准（Stable Diffusion、LLaMA 等都用 LoRA）
>
> **后续**：QLoRA（4-bit）、LoRA+（不对称）、DoRA（幅度+方向）
>
> **面试核心**：
> - Q: 为什么零延迟？A: $W_0 + BA$ 预先合并
> - Q: 为什么 r=1 够？A: ΔW 低秩，子空间分析证明
> - Q: 为什么 Q+V？A: Q="关注什么"，V="传递什么"
