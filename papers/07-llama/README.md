# 📖 LLaMA: Open and Efficient Foundation Language Models

> **论文**：Touvron et al., 2023 (Meta AI) | arXiv 2023
>
> **一句话总结**：用公开数据训练的 7B-65B 开源模型——LLaMA-13B 超越 GPT-3 (175B)，LLaMA-65B 竞争 Chinchilla-70B 和 PaLM-540B。

---

# 第一层：鸟瞰

## 🎯 核心贡献

1. **开源民主化**：首个用**纯公开数据**训练、性能接近闭源 SOTA 的模型，向研究社区开放
2. **Chinchilla 验证**：LLaMA-13B (10x 更小) 超越 GPT-3 (175B)——直接验证了"小模型+多数据"的 Chinchilla 思想
3. **推理优先**：不只是训练最优，而是**推理最优**——更小模型部署成本更低
4. **架构改进总结**：Pre-RMSNorm、SwiGLU、RoPE、因果多查询注意力——集成了当时最佳实践

## 📍 知识网络定位

```
Chinchilla (2022.03) → 70B/1.4T（计算最优 = 小模型+多数据）
         ↓
   【LLaMA (2023.02)】→ 7B-65B，公开数据，开源
         ↓
   Alpaca (2023.03) → LLaMA + 指令微调
   LLaMA-2 (2023.07) → 7B-70B，RLHF，商用许可
   Mistral 7B (2023.09) → 继承 LLaMA 架构，更高效
   LLaMA-3 (2024.04) → 8B-70B，更大词表，多语言
```

---

# 第二层：精读

## 1. 引言：为什么做 LLaMA？

### 核心论点

> "The preferred model is not the fastest to train but the fastest at inference."

Chinchilla 关注训练最优，LLaMA 关注**推理最优**——在目标性能下，选最小的模型。

> ❓ **为什么推理更重要？** 模型训练一次，但推理无数次。部署成本 = 推理成本 × 调用量。13B 模型的推理成本只有 175B 的 1/13。

### 三个突破

1. **LLaMA-13B > GPT-3 (175B)**：10x 更小但更好
2. **纯公开数据**：不依赖任何私有数据（GPT-3 用了 WebText2，Chinchilla 用了未公开数据）
3. **开源**：所有模型权重向研究社区开放

## 2. 方法

### 2.1 训练数据

| 数据源 | 比例 | Epochs | 大小 |
|--------|------|--------|------|
| CommonCrawl | 67% | 1.10 | 3.3TB |
| C4 | 15% | 1.06 | 783GB |
| GitHub | 4.5% | 0.64 | 328GB |
| Wikipedia | 4.5% | 2.45 | 83GB |
| Books | 4.5% | 2.23 | 85GB |
| ArXiv | 2.5% | 1.06 | 92GB |
| StackExchange | 2% | 1.03 | 78GB |

**总量**：~1.4T tokens（部分模型用 1T）

> 💡 **和 GPT-3 数据对比**：GPT-3 用了 300B tokens（~4x 更少），且包含私有数据（WebText2）。LLaMA 用了 1.4T tokens 的**纯公开数据**。

> ❓ **数据质量**：LLaMA 对 CommonCrawl 做了严格过滤——CCNet pipeline 去重、语言识别、质量过滤、Wikipedia 参考分类。这比 GPT-3 的过滤更严格。

### 2.2 架构改进

LLaMA 集成了 4 个关键改进（每个都来自前人的工作）：

| 改进 | 来源 | 为什么更好 |
|------|------|-----------|
| **Pre-RMSNorm** | GPT-2 + Gopher | 比 LayerNorm 更稳定、计算更少 |
| **SwiGLU** | PaLM | 比 ReLU/GeLU 激活函数效果更好 |
| **RoPE** | GPT-NeoX | 旋转位置编码，支持相对位置，外推性好 |
| **因果多查询注意力** | PaLM | KV cache 只需存 1 份而非 n_head 份，推理更快 |

#### SwiGLU 详解

标准 FFN：$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x)$

SwiGLU：$\text{FFN}(x) = (W_2 x) \cdot \text{SiLU}(W_1 x) \otimes (W_3 x)$

- SiLU (Swish) 激活函数：$\sigma(x) = x \cdot \text{sigmoid}(x)$
- 门控机制（GLU）：额外的 $W_3$ 做门控
- 缺点：需要 3 个权重矩阵（而非 2 个），参数量增加 ~50%

> ❓ **为什么 SwiGLU 比 ReLU 好？** 实验证明门控 + 平滑激活比硬截断更好。PaLM、LLaMA、Mistral 都用了 SwiGLU。

#### RoPE（旋转位置编码）详解

传统位置编码：绝对位置，添加到输入中。问题：不能自然地表达相对位置关系。

RoPE：把位置信息编码为旋转矩阵，在注意力计算中自然包含相对位置。

$$q_m \cdot k_n = |q||k| \cos(m\theta - n\theta)$$

位置 m 的 query 和位置 n 的 key 的内积只依赖**相对位置** (m-n)。

> 💡 RoPE 后来成为了大模型的标准选择——LLaMA、Mistral、Qwen、DeepSeek 全部使用 RoPE。

#### 因果多查询注意力（GQA）

标准多头注意力：每个 head 有独立的 K、V → KV cache 大小 = n_layers × n_heads × d_head

多查询注意力（MQA）：所有 head 共享 K、V → KV cache 大小 = n_layers × 1 × d_head

> 💡 KV cache 大小直接决定推理时需要的显存。MQA 将 KV cache 缩小 ~n_head 倍（LLaMA-65B 从 64x 缩到 1x）——推理速度和成本大幅降低。

### 2.3 训练配置

| 模型 | 参数 | 层数 | 维度 | 头数 | LR | Tokens |
|------|------|------|------|------|-----|--------|
| 7B | 6.7B | 32 | 4096 | 32 | 3e-4 | 1.0T |
| 13B | 13B | 40 | 5120 | 40 | 3e-4 | 1.0T |
| 32B | 32.5B | 60 | 6656 | 52 | 1.5e-4 | 1.4T |
| 65B | 65.2B | 80 | 8192 | 64 | 1.5e-4 | 1.4T |

> 💡 7B 和 13B 用了 1T tokens（~1 epoch），32B 和 65B 用了 1.4T tokens（~1 epoch）。这和 Chinchilla 的"多数据"策略一致。

## 3. 实验结果

### 核心对比

| 模型 | 参数 | Tokens | MMLU | 推理成本 |
|------|------|--------|------|---------|
| GPT-3 | 175B | 300B | ~43.9% | 基准 |
| Chinchilla | 70B | 1.4T | 67.5% | 0.4x |
| PaLM | 540B | 780B | ~57% | 3x |
| **LLaMA-13B** | **13B** | **1.0T** | **~47%** | **0.07x** |
| **LLaMA-65B** | **65B** | **1.4T** | **~63.4%** | **0.37x** |

> 💡 **LLaMA-13B 在多数基准上超越了 GPT-3**——尽管只有 GPT-3 参数的 1/13。直接验证了 Chinchilla 的预言。

> 💡 **LLaMA-65B 接近 Chinchilla-70B**——但 LLaMA 用的是纯公开数据，Chinchilla 用了未公开的数据。

### 具体任务

- **常识推理**：LLaMA-65B 在 PIQA、SIQA、HellaSwag 上接近或超过 PaLM-540B
- **阅读理解**：LLaMA-65B 在 RACE、SQuAD 上有竞争力
- **数学**：LLaMA-65B 在 MATH、GSM8K 上不如 Minerva（但 Minerva 在数学数据上做了额外训练）
- **代码**：LLaMA-65B 在 HumanEval 上不如 Codex（但 LLaMA 没有用大量代码数据）

---

# 第三层：批判性思考

## 🤔 设计决策分析

### 为什么不用更多 epochs？

LLaMA 对大部分数据只训练了 ~1 epoch（Wikipedia 和 Books 约 2 epochs）。

> ❓ **批判**：Chinchilla 证明了数据量很重要，但"更多数据"不等于"更多 epochs"。1 epoch 意味着模型只看一次数据——不会有记忆问题，但也可能浪费了数据中的信息。后来 LLaMA-2 用了 2T tokens（更多数据源），效果进一步提升。

### GQA 的代价

多查询注意力节省了推理成本，但也可能限制了模型的表达能力。

> ❓ **后来 LLaMA-2 的选择**：LLaMA-2 改回了 GQA（分组查询注意力，介于 MHA 和 MQA 之间）——说明纯 MQA 可能确实有性能代价。

### 数据偏差

LLaMA 主要用英文数据——CommonCrawl 67% + C4 15% + GitHub 4.5% 几乎全是英文。

> ❓ **批判**：多语言能力有限。后来 LLaMA-3 大幅增加了多语言数据。

## ⚠️ 局限性

1. **只有预训练**：不做 RLHF/指令微调——纯 base model，对话能力有限
2. **英文为主**：多语言能力不如 mT5/BLOOM
3. **代码能力弱**：GitHub 数据只有 4.5%，不如 Codex
4. **数学弱**：没有专门的数学训练
5. **偏见/毒性**：论文承认训练数据中的偏见被模型继承

## 🎯 面试视角

**Q1: LLaMA 的核心创新是什么？**

> A: 不是单一技术创新，而是**工程整合**——把 Pre-RMSNorm、SwiGLU、RoPE、MQA 四个已知改进组合起来，用纯公开数据按 Chinchilla 思想训练，开源。关键是"开源+公开数据+推理优先"。

**Q2: 为什么 LLaMA-13B 能超过 GPT-3 (175B)？**

> A: 三个原因：(1) 用了 3x 更多数据（1T vs 300B tokens），验证 Chinchilla；(2) 架构改进（SwiGLU、RoPE 等）；(3) 更好的数据质量（严格的 CommonCrawl 过滤）。

**Q3: RoPE 和传统位置编码有什么区别？**

> A: 传统位置编码是绝对的（token 位置 1、2、3...），RoPE 通过旋转矩阵编码相对位置——query 和 key 的内积自然包含相对位置信息。优势：更好的长度外推性。

**Q4: 为什么推理成本重要？**

> A: 模型训练一次但推理无数次。13B 模型的推理成本只有 175B 的 1/13——在 API 部署中意味着更低的价格和更快的响应。这也是"推理优先"策略的核心。

**Q5: LLaMA 对大模型生态有什么影响？**

> A: LLaMA 开创了"开源基础模型"时代。Alpaca、Vicuna、Koala 等数十个微调模型都基于 LLaMA。后来的 LLaMA-2、Mistral、Qwen 等都继承了 LLaMA 的架构选择。

---

# 第四层：知识网络

## 📅 时间线

```
Chinchilla (2022.03) → 计算最优 = 小模型+多数据
    【LLaMA (2023.02)】→ 开源验证 Chinchilla
Alpaca (2023.03) → LLaMA + Stanford 指令微调
Vicuna (2023.03) → LLaMA + 对话微调
LLaMA-2 (2023.07) → 7B-70B，RLHF，商用
Mistral 7B (2023.09) → 继承 LLaMA 架构，更高效
LLaMA-3 (2024.04) → 8B-70B，更大词表
```

## ❓ 深度思考题

1. LLaMA 用公开数据就能接近闭源 SOTA——这说明"秘密武器"是数据质量还是模型架构？
2. 如果 LLaMA 用了 GPT-3 的训练数据（包括 WebText2），效果会更好吗？
3. RoPE 的相对位置编码为什么比绝对位置编码更好？在什么场景下绝对编码反而有优势？

## 📚 延伸阅读

| 论文 | 年份 | 关系 |
|------|------|------|
| Chinchilla | 2022 | LLaMA 的理论基础 |
| LLaMA-2 | 2023 | 直接续作 |
| Mistral 7B | 2023 | 继承架构 |
| Alpaca | 2023 | LLaMA + 指令微调 |
