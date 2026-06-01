# LLaMA — 精读融合版

> 论文原文保持不动，穿插中文讲解（📖 标记）。

---

# Abstract

We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively [...] LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B.

> 📖 **讲解 · 鸟瞰**
>
> **一句话**：纯公开数据 + 开源 + 小模型+多数据 = 接近闭源 SOTA。
>
> **核心数字**：13B > GPT-3 (10x 更小)；65B 竞争 540B (8x 更小)。

---

# 1. Introduction

[原文...]

> 📖 **讲解 · 精读 · 核心论点**
>
> **推理优先**："The preferred model is not the fastest to train but the fastest at inference."
>
> **vs Chinchilla**：Chinchilla 关注训练最优，LLaMA 关注推理最优——13B 部署成本只有 175B 的 1/13。
>
> **开源的原因**：Meta 的商业模式不依赖卖模型 API，开源能吸引生态。

---

# 2. Approach

## 2.1 Pre-training Data

Table 1: 训练数据

> 📖 **讲解 · 精读 · 训练数据**
>
> | 特点 | LLaMA | GPT-3 |
> |------|-------|-------|
> | 总 tokens | 1.0-1.4T | 300B |
> | 私有数据 | 无 | WebText2 |
> | 过滤 | CCNet 严格 | 简单 |
>
> **大部分 1 epoch**：避免记忆。Wikipedia 和 Books 约 2 epochs（高质量多看几遍）。
>
> **CCNet pipeline**：去重 → 语言识别 → 质量过滤 → Wikipedia 参考分类。比 GPT-3 的过滤更严格。

## 2.2 Architecture

[原文...]

> 📖 **讲解 · 精读 · 四个架构改进**
>
> | 改进 | 来自 | 作用 |
> |------|------|------|
> | Pre-RMSNorm | GPT-2 + Gopher | 训练更稳定，比 LayerNorm 快 ~7% |
> | SwiGLU | PaLM | 门控+平滑激活 > ReLU 硬截断 |
> | RoPE | GPT-NeoX | 旋转位置编码，相对位置，外推性好 |
> | 因果 MQA | PaLM | KV cache 缩小 n_head 倍 |
>
> **RoPE 核心公式**：$q_m \cdot k_n = |q||k|\cos((m-n)\theta)$ → 内积只依赖相对位置 (m-n)
>
> **MQA**：所有 head 共享 K,V → LLaMA-65B 的 KV cache 缩小 64 倍
>
> **SwiGLU**：$\text{FFN}(x) = (W_2 x) \cdot \text{SiLU}(W_1 x) \otimes (W_3 x)$ → 3 个矩阵，FFN 维度调整为 $\frac{2}{3} \times 4 \times d_{model}$ 补偿
>
> ❓ **为什么这些成为标准？** 不是因为这些被证明是"最优"的，而是因为它们在 LLaMA 中效果好 + 开源可复现 → 后来所有人都跟着用。

## 2.3 Optimizer

[原文...]

> 📖 **讲解 · 精读 · 训练配置**
>
> - AdamW, β1=0.9, β2=0.95
> - Cosine LR schedule，warmup 2000 steps
> - Weight decay = 0.1
> - Gradient clipping = 1.0
> - 65B: 2048 A100 × 21 天 = ~43K A100 天

---

# 3. Main Results

## 3.1 Common Sense Reasoning

Table 3: 常识推理

> 📖 **讲解 · 精读 · 常识推理**
>
> **LLaMA-65B 在 6/8 基准上超越 PaLM-540B**——参数只有 1/8
>
> **LLaMA-13B 在多数基准上超越 GPT-3**——参数只有 1/13
>
> ❓ **BoolQ 为什么不如 PaLM？** BoolQ 需要知识——更大模型存更多知识。

## 3.4 Mathematical Reasoning

Table 7: GSM8k

> 📖 **讲解 · 精读 · 数学推理**
>
> **LLaMA-65B maj1@100 = 69.7 > Minerva-62B = 68.5**——虽然 Minerva 在数学数据上专门微调
>
> 但 pass@1 (50.9) 仍不如 PaLM-540B (56.5)——大模型单样本推理仍有优势

---

> 📖 **讲解 · 知识网络**
>
> **LLaMA 的遗产**：开创"开源基础模型"时代。RoPE+SwiGLU+Pre-RMSNorm 成为大模型标准。
>
> **后续发展**：LLaMA-2 (GQA+RLHF) → LLaMA-3 (15T tokens) → 整个开源 LLM 生态
>
> **面试核心**：
> - Q: LLaMA 为什么能超过 GPT-3？A: 3x 数据 + 架构改进 + 数据质量
> - Q: RoPE？A: 旋转位置编码，$q_m \cdot k_n$ 只依赖 (m-n)
> - Q: MQA？A: 共享 K,V，KV cache 缩小 64 倍
