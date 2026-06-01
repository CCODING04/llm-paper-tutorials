# 📖 LLaMA: Open and Efficient Foundation Language Models

> **论文**：Touvron et al., 2023 (Meta AI) | arXiv 2023
>
> **一句话总结**：用纯公开数据训练的 7B-65B 开源模型——LLaMA-13B 超越 GPT-3 (175B)，LLaMA-65B 竞争 Chinchilla-70B 和 PaLM-540B。

---

# 第一层：鸟瞰

## 🎯 核心贡献

1. **开源民主化**：首个用纯公开数据训练、性能接近闭源 SOTA 的开源模型——改变了 LLM 生态
2. **Chinchilla 验证**：LLaMA-13B (10x 更小) 超越 GPT-3 (175B)——"小模型+多数据"策略的开源验证
3. **推理优先**：不只训练最优，而是推理最优——更小模型部署成本更低
4. **架构改进集成**：Pre-RMSNorm、SwiGLU、RoPE、因果 MQA——四个已知改进的最佳实践组合
5. **训练细节完全公开**：数据混合、训练配置、训练效率——为社区提供了可复现的参考

## 📍 知识网络定位

```
Chinchilla (2022.03) → N∝C^0.50（小模型+多数据）
GPT-3 (2020) → 175B, 闭源, 300B tokens
PaLM (2022.04) → 540B, 闭源
         ↓
   【LLaMA (2023.02)】→ 7B-65B, 纯公开数据, 开源
         ↓
   Alpaca (2023.03) → LLaMA + Stanford 指令微调
   Vicuna (2023.03) → LLaMA + 对话微调
   LLaMA-2 (2023.07) → 7B-70B, RLHF, 商用许可
   Mistral 7B (2023.09) → 继承架构，更高效
   LLaMA-3 (2024.04) → 8B-70B, 15T tokens, 更大词表
```

**关键对比**：
- **vs GPT-3**：LLaMA-13B (10x 更小) 在多数基准上超越 GPT-3
- **vs Chinchilla**：LLaMA-65B 接近 Chinchilla-70B——但 LLaMA 用纯公开数据
- **vs PaLM**：LLaMA-65B 在常识推理上超越 PaLM-540B（BoolQ 85.3 vs 88.0 除外）

---

# 第二层：精读

## 1. 引言：为什么做 LLaMA？

### 核心论点

> "The preferred model is not the fastest to train but the fastest at inference."

**推理优先** vs Chinchilla 的训练优先：
- Chinchilla：给定计算预算，找到训练最优的 (N, D)
- LLaMA：在目标性能下，选最小的模型（推理最快）

> ❓ **为什么推理更重要？** 模型训练一次，推理无数次。13B 推理成本只有 175B 的 1/13。在 API 部署中 = 更低价格 + 更快响应。

### 为什么开源？

> "We release all our models to the research community."

因为 Meta 的商业模式不依赖模型本身（不像 OpenAI 卖 API），开源能吸引生态、加速研究。

## 2. 方法

### 2.1 训练数据

| 数据源 | 采样比例 | Epochs | 磁盘大小 |
|--------|---------|--------|---------|
| CommonCrawl | 67.0% | 1.10 | 3.3TB |
| C4 | 15.0% | 1.06 | 783GB |
| GitHub | 4.5% | 0.64 | 328GB |
| Wikipedia | 4.5% | 2.45 | 83GB |
| Books (Gutenberg + Books3) | 4.5% | 2.23 | 85GB |
| ArXiv | 2.5% | 1.06 | 92GB |
| StackExchange | 2.0% | 1.03 | 78GB |

**总量**：~1.4T tokens（7B/13B 用 1T，33B/65B 用 1.4T）

> 📖 **讲解 · 和 GPT-3 数据的关键区别**
>
> | | GPT-3 | LLaMA |
> |---|-------|-------|
> | 总 tokens | 300B | 1.0T~1.4T |
> | 私有数据 | WebText2 (3.4x 过采样) | **无** |
> | CommonCrawl 过滤 | 简单 | CCNet pipeline（严格） |
> | GitHub | 无 | 4.5% (328GB) |
>
> LLaMA 的数据量是 GPT-3 的 3-4.7 倍，且全是公开数据。

> ❓ **为什么大部分数据只训练 1 epoch？** 避免记忆。但这也意味着模型只看一次数据——后来 LLaMA-2 用了更多数据源（2T tokens），LLaMA-3 用了 15T tokens。

### 2.2 架构改进

LLaMA 集成了 4 个已知改进——不是新发明，而是最佳实践的组合：

#### 改进 1: Pre-RMSNorm（替代 Post-LayerNorm）

**标准 Transformer**: $x_{out} = \text{LayerNorm}(x + \text{Sublayer}(x))$ (Post-LN)

**LLaMA**: $x_{out} = x + \text{Sublayer}(\text{RMSNorm}(x))$ (Pre-RMSNorm)

- RMSNorm: $\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2}} \cdot g$（不计算均值，比 LayerNorm 快 ~7%）
- Pre-Norm: 归一化在子层之前 → 梯度流更稳定（和 GPT-2 用的 Pre-LN 同理）

> ❓ **为什么 Pre > Post？** Post-LN 在训练初期梯度可能爆炸（深层梯度 >> 浅层），Pre-Norm 的梯度从深层到浅层几乎不变（梯度高速公路）。

#### 改进 2: SwiGLU（替代 ReLU/GeLU）

**标准 FFN**: $\text{FFN}(x) = W_2 \cdot \sigma(W_1 x)$

**SwiGLU**: $\text{FFN}(x) = (W_2 x) \cdot \text{SiLU}(W_1 x) \otimes (W_3 x)$

- SiLU (Swish): $\sigma(x) = x \cdot \text{sigmoid}(x)$——平滑激活，不像 ReLU 硬截断
- GLU (门控): 额外的 $W_3$ 做门控——选择性传递信息
- **代价**: 3 个权重矩阵 vs 2 个 → 参数量增加 ~50%。LLaMA 通过调整 $d_{ffn} = \frac{2}{3} \cdot 4 \cdot d_{model}$ 来补偿。

> ❓ **为什么 SwiGLU 比 ReLU 好？** PaLM 的实验证明门控+平滑激活 > 硬截断。后来的 Mistral、Qwen、DeepSeek 都用了 SwiGLU。

#### 改进 3: RoPE（旋转位置编码）

**传统位置编码**: 绝对位置 1, 2, 3... → 添加到 embedding 中

**RoPE**: 把位置编码为旋转矩阵 → 内积自然包含相对位置

$$q_m \cdot k_n = |q||k| \cos((m-n)\theta)$$

位置 m 的 query 和位置 n 的 key 的内积只依赖相对位置 (m-n)。

> ❓ **RoPE 的优势？** (1) 自然表达相对位置；(2) 长度外推性更好（可通过 NTK-aware scaling 扩展到更长序列）；(3) 后来成为大模型的标准选择。

#### 改进 4: 因果多查询注意力（Causal MQA）

**标准 MHA**: 每个 head 有独立的 K, V → KV cache = $n_{layers} \times n_{heads} \times d_{head}$

**MQA**: 所有 head 共享 K, V → KV cache = $n_{layers} \times 1 \times d_{head}$

> 💡 **KV cache 大小直接决定推理显存**。LLaMA-65B 有 64 个 head → MQA 把 KV cache 缩小 64 倍。推理速度和成本大幅降低。

> ❓ **MQA 的代价？** 表达能力可能受限。LLaMA-2 改为 GQA（分组查询注意力，介于 MHA 和 MQA 之间），说明纯 MQA 确实有性能代价。

### 2.3 模型配置

| 模型 | 参数 | 层数 | $d_{model}$ | Heads | LR | Tokens |
|------|------|------|-----------|-------|-----|--------|
| 7B | 6.7B | 32 | 4096 | 32 | 3e-4 | 1.0T |
| 13B | 13.0B | 40 | 5120 | 40 | 3e-4 | 1.0T |
| 33B | 32.5B | 60 | 6656 | 52 | 1.5e-4 | 1.4T |
| 65B | 65.2B | 80 | 8192 | 64 | 1.5e-4 | 1.4T |

- Batch size: 4M tokens（所有模型）
- 序列长度: 2048 tokens
- 训练效率: 65B 在 2048 A100 上 ~380 tokens/sec/GPU，训练 21 天

### 2.4 高效实现

- **激活重计算**: 手动实现 transformer 层的 backward → 减少内存
- **模型并行 + 序列并行**: 分布式训练
- **通信重叠**: 计算和 GPU 间通信并行

## 3. 实验结果

### 3.1 常识推理（Common Sense Reasoning）

| 模型 | BoolQ | PIQA | HellaSwag | WinoGrande |
|------|-------|------|-----------|-----------|
| GPT-3 175B | 60.5 | 81.0 | 78.9 | 70.2 |
| Chinchilla 70B | 83.7 | 81.8 | 80.8 | 74.9 |
| PaLM 540B | 88.0 | 82.3 | 83.4 | 81.1 |
| **LLaMA 13B** | 78.1 | 80.1 | 79.2 | 73.0 |
| **LLaMA 65B** | **85.3** | **82.8** | **84.2** | 77.0 |

> 💡 **LLaMA-65B 在 6/8 常识推理基准上超越 PaLM-540B**（参数只有 1/8！）
>
> **LLaMA-13B 在多数基准上超越 GPT-3**（参数只有 1/13！）

> ❓ **为什么 BoolQ 上不如 PaLM-540B？** BoolQ 是布尔问答，需要更多知识——更大的模型存储了更多知识。

### 3.2 问答（QA）

| 模型 | NQ 64-shot | TriviaQA 64-shot |
|------|-----------|-----------------|
| GPT-3 175B | 29.9 | - |
| Chinchilla 70B | 35.5 | 64.6 |
| **LLaMA 65B** | **39.9** | **73.0** |

> 💡 LLaMA-65B 在问答上 SOTA——更多数据 = 更多知识 = 更好的问答。

### 3.3 阅读理解

| 模型 | RACE-middle | RACE-high |
|------|-----------|-----------|
| GPT-3 175B | 58.4 | 45.5 |
| PaLM 540B | 68.1 | 49.1 |
| **LLaMA 65B** | **67.9** | **51.6** |

> 💡 RACE-high 上 LLaMA-65B 超越 PaLM-540B (51.6 vs 49.1)——更多数据让理解能力更强。

### 3.4 数学推理

| 模型 | GSM8k maj1@100 |
|------|---------------|
| PaLM 540B | 56.5 |
| Minerva 62B | 68.5 |
| **LLaMA 65B** | **69.7** |

> 💡 LLaMA-65B 在 GSM8k 上超过 Minerva-62B——虽然 Minerva 在数学数据上做了专门微调！这说明更多通用数据也能提升数学能力。

### 3.5 代码生成

| 模型 | HumanEval pass@1 |
|------|-----------------|
| PaLM 62B | - |
| **LLaMA 65B** | **23.7** |

> ❓ **为什么 LLaMA 的代码能力不如 Codex？** Codex 在大量 GitHub 代码上专门训练。LLaMA 只有 4.5% 的 GitHub 数据。但作为通用模型，LLaMA 的代码能力已经很有竞争力。

### 3.6 MMLU

| 模型 | 5-shot |
|------|--------|
| GPT-3 175B | 43.9 |
| Chinchilla 70B | 67.5 |
| PaLM 540B | ~57 |
| **LLaMA 65B** | **63.4** |

> ❓ **为什么 MMLU 不如 Chinchilla？** Chinchilla 的 1.4T tokens 中可能有更多高质量知识数据。LLaMA 用了纯公开数据——数据质量可能不如 Chinchilla 的 MassiveText。

---

# 第三层：批判性思考

## 🤔 设计决策分析

### 为什么不用更多 epochs？

> ❓ **Chinchilla 验证了"多数据"但没说"多 epochs"**。LLaMA 大部分数据只训练 1 epoch。后来 LLaMA-2 用了 2T tokens，LLaMA-3 用了 15T tokens——但都是不同的数据源，不是在相同数据上多跑几遍。

### GQA vs MQA

LLaMA 用了纯 MQA（所有 head 共享 K,V）。LLaMA-2 改为 GQA（分组共享）。

> ❓ **为什么改？** 纯 MQA 可能在某些任务上有性能损失。GQA 是 MHA 和 MQA 的折中——比如 64 个 head 分成 8 组，每组 8 个 head 共享 K,V。

### 数据偏差

CommonCrawl 67% + C4 15% + GitHub 4.5% = 几乎全是英文。

> ❓ **多语言能力有限**。LLaMA-3 大幅增加了多语言数据。

### 训练成本

65B: 2048 A100 × 21 天 = ~43,000 A100 天

> ❓ **对研究机构来说仍然昂贵**。但远低于训练 540B 模型的成本。

## ⚠️ 局限性

1. **只有预训练**：不做 RLHF/指令微调——纯 base model
2. **英文为主**：多语言能力不如 mT5/BLOOM
3. **代码/数学弱**：不如专门模型（Codex/Minerva）
4. **偏见/毒性**：论文承认训练数据中的偏见被继承
5. **序列长度 2048**：比 PaLM (2048) 相同，但比后来模型短

## 🎯 面试视角

**Q1: LLaMA 的核心创新是什么？**

> A: 不是单一技术创新，而是工程整合——把 Pre-RMSNorm、SwiGLU、RoPE、MQA 四个已知改进组合，用纯公开数据按 Chinchilla 思想训练，然后开源。关键是"开源+公开数据+推理优先"。

**Q2: 为什么 LLaMA-13B 能超过 GPT-3 (175B)？**

> A: 三个原因：(1) 3x 更多数据（1T vs 300B tokens），验证 Chinchilla；(2) 架构改进（SwiGLU、RoPE、MQA 等）；(3) 更好的数据质量（严格的 CCNet 过滤）。

**Q3: RoPE 和传统位置编码有什么区别？**

> A: 传统位置编码是绝对的（位置 1,2,3...），RoPE 通过旋转矩阵编码相对位置——query 和 key 的内积 $q_m \cdot k_n = |q||k|\cos((m-n)\theta)$ 自然包含相对位置信息。优势：更好的长度外推性。

**Q4: MQA 为什么能加速推理？**

> A: 标准多头注意力每个 head 有独立的 K,V，推理时 KV cache = $n_{layers} \times n_{heads} \times d_{head}$。MQA 让所有 head 共享一份 K,V → KV cache 缩小 $n_{heads}$ 倍（LLaMA-65B: 64 倍）→ 推理显存大幅降低。

**Q5: LLaMA 对大模型生态的影响？**

> A: 开创了"开源基础模型"时代。Alpaca、Vicuna、Koala 等数十个微调模型都基于 LLaMA。后来的 LLaMA-2、Mistral、Qwen 都继承了 LLaMA 的架构选择（RoPE+SwiGLU+Pre-RMSNorm 成为大模型标准）。

---

# 第四层：知识网络

## 📅 时间线

```
Chinchilla (2022.03) → 计算最优 = 小模型+多数据
GPT-3 (2020.05) → 175B 闭源
PaLM (2022.04) → 540B 闭源
    【LLaMA (2023.02)】→ 开源验证 Chinchilla
Alpaca (2023.03) → LLaMA + Stanford 指令微调
Vicuna (2023.03) → LLaMA + 对话微调
LLaMA-2 (2023.07) → 7B-70B, RLHF, 商用
Mistral 7B (2023.09) → 继承架构，更高效
LLaMA-3 (2024.04) → 8B-70B, 15T tokens
```

## ❓ 深度思考题

1. **概念题**：LLaMA 没有发明任何新架构组件——为什么它能成为里程碑？工程的"集成创新"和学术的"原创创新"哪个更重要？
2. **设计题**：如果训练 LLaMA 但用 MQA 改为 GQA（如 LLaMA-2），你会怎么分组？为什么？
3. **批判题**：LLaMA 用公开数据就能接近闭源 SOTA——这说明"秘密武器"是数据质量还是模型架构？还是只是因为 Chinchilla 已经证明了方向？
4. **扩展题**：LLaMA 的架构选择（RoPE+SwiGLU+Pre-RMSNorm+MQA）后来成为了标准。有没有可能这些选择只是"够好"而非"最优"？什么证据能证明它们是最优的？
5. **实践题**：LLaMA-65B 训练用了 2048 A100 × 21 天。如果只有 256 A100，你会怎么调整策略？

## 📚 延伸阅读

| 论文 | 年份 | 关系 |
|------|------|------|
| Chinchilla | 2022 | LLaMA 的理论基础 |
| LLaMA-2 | 2023 | 直接续作，GQA+RLHF |
| Mistral 7B | 2023 | 继承架构，更高效 |
| Alpaca | 2023 | LLaMA + 指令微调 |
| PaLM | 2022 | 竞争对手（SwiGLU 和 MQA 的来源） |
