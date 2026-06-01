# LLaMA: Open and Efficient Foundation Language Models — 精读融合版

> 阅读指南：论文原文保持不动，每个章节后穿插中文讲解（📖 标记）。

---

# Abstract

We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively [...] LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B.

> 📖 **讲解 · 鸟瞰**
>
> **一句话**：公开数据 + 开源 + 小模型+多数据 = 接近闭源 SOTA。
>
> **三个突破**：LLaMA-13B > GPT-3；纯公开数据；开源权重。

---

# 1. Introduction

[论文原文...]

> 📖 **讲解 · 精读 · 核心论点**
>
> LLaMA 的立场：**推理优先**。"The preferred model is not the fastest to train but the fastest at inference."
>
> 和 Chinchilla 的区别：Chinchilla 关注训练最优，LLaMA 关注推理最优——13B 模型部署成本只有 175B 的 1/13。

# 2. Approach

## 2.1 Pre-training Data

[论文原文：Table 1 数据混合...]

> 📖 **讲解 · 精读 · 训练数据**
>
> **核心特点**：
> 1. **纯公开数据**——CommonCrawl、C4、GitHub、Wikipedia、Books、ArXiv、StackExchange
> 2. **~1.4T tokens**——比 GPT-3 的 300B 多 4.7x
> 3. **大部分 1 epoch**——避免记忆
> 4. **严格质量过滤**——CCNet pipeline
>
> **和 GPT-3 对比**：GPT-3 用了私有 WebText2（3.4x 过采样），LLaMA 全公开。

## 2.2 Architecture

[论文原文：四个架构改进...]

> 📖 **讲解 · 精读 · 架构改进**
>
> | 改进 | 来自 | 作用 |
> |------|------|------|
> | Pre-RMSNorm | GPT-2 | 训练更稳定 |
> | SwiGLU | PaLM | 更好的激活函数 |
> | RoPE | GPT-NeoX | 相对位置编码 |
> | 因果 MQA | PaLM | KV cache 缩小 n_head 倍 |
>
> **RoPE 详解**：旋转位置编码，内积自然包含相对位置：(m-n)θ
>
> **MQA 详解**：所有头共享 K、V → KV cache 从 n_head 份缩到 1份 → 推理更快

# 3. Results

[论文原文...]

> 📖 **讲解 · 精读 · 核心结果**
>
> **LLaMA-13B vs GPT-3**：13B 在多数基准上超越 175B——验证 Chinchilla
>
> **LLaMA-65B**：接近 Chinchilla-70B 和 PaLM-540B——但推理成本只有 PaLM 的 1/8
>
> ❓ **局限性**：数学和代码不如专门模型（Minerva/Codex），因为训练数据中代码/数学占比少

---

> 📖 **讲解 · 知识网络**
>
> **LLaMA 的遗产**：
> - 开创"开源基础模型"时代
> - 架构成为标准（RoPE + SwiGLU + Pre-RMSNorm）
> - Alpaca/Vicuna/Mistral 等都基于 LLaMA
>
> **面试核心**：
> - Q: LLaMA 为什么能超过 GPT-3？A: 3x 更多数据 + 架构改进 + 更好的数据质量
> - Q: RoPE？A: 旋转位置编码，相对位置，内积含 (m-n)θ
> - Q: 为什么开源？A: 民主化 LLM 研究，降低进入门槛
