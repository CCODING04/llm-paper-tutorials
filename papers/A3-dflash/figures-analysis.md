# 📊 DFlash 图表分析

> 论文共包含 **5 个 Figure** 和 **13 个 Table**，本文档逐一分析。

---

## Figure 1: Speedup Comparison — DFlash vs EAGLE-3 vs Baseline

![Figure 1: Speedup comparison](./images/174e35fbc7f5152d2dd3971c146494439009357e31c4f7c5a27fd89a607f497c.jpg)

**位置**：Section 1 Introduction
**Caption**：Speedup comparison between DFlash, EAGLE-3 against Autoregressive Decoding on Qwen3-8B with the Transformers backend. Overall, DFlash achieves more than 2.5× higher speedup than EAGLE-3.

### 客观描述（mimo-v2.5 多模态提取）
- **类型**：分组柱状图，对比 3 种方法在 7 个推理任务上的加速比
- **X 轴**：7 个 benchmark（GSM8K, Math500, AIME25, HumanEval, MBPP, LiveCodeBench, MT-Bench）
- **Y 轴**：Speedup (×)，范围 0~6
- **三组柱**：
  - 灰色柱：Baseline（加速比固定为 1.00×）
  - 绿色柱：EAGLE-3（~1.81-2.23×）
  - 蓝色柱：DFlash（~2.75-6.08×）

### 关键数据（从图中提取）

| Benchmark | EAGLE-3 | DFlash | DFlash/EAGLE-3 倍率 |
|-----------|---------|--------|---------------------|
| GSM8K | 2.23× | 5.15× | 2.31× |
| Math500 | 2.05× | 6.08× | 2.97× |
| AIME25 | 2.05× | 5.62× | 2.74× |
| HumanEval | 2.17× | 5.14× | 2.37× |
| MBPP | 1.93× | 4.65× | 2.41× |
| LiveCodeBench | 1.81× | 5.51× | 3.04× |
| MT-Bench | 1.90× | 2.75× | 1.45× |

### 深度分析

- **独立解读**：DFlash 在所有任务上都远超 EAGLE-3。差距在 Math/Code 任务最大（~2.5-3×），Chat 任务（MT-Bench）差距最小（1.45×）。DFlash 在 Math500 上达到最高加速 6.08×。

- **对照 caption**：一致。Caption 说 "more than 2.5× higher speedup than EAGLE-3"。实际倍率范围 1.45×-3.04×，平均约 2.5×。

- **验证的假设**：支持核心 claim——block diffusion drafting 比自回归 drafting 快得多。特别地，Math/Code 任务规律性强，draft 模型预测更准，加速更大。

- **批判性评价**：
  - ✅ Baseline=1.00× 公平（同一模型，无加速）
  - ✅ EAGLE-3 在论文 Table 1 中有更多配置对比（tree size 16/60）
  - ⚠️ 这只是 Qwen3-8B 一个模型的结果，通用性需看 Table 5（LLaMA）和 Table 11（更多模型）
  - ⚠️ Y 轴从 0 开始，柱状图没有视觉误导

- **面试价值**：这是论文的"门面图"。一句话——DFlash 在所有任务上全面碾压 EAGLE-3，Math/Code 最高 6×，Chat 最低也有 2.75×。

---

## Figure 2: DFlash Inference Design — 推理架构图

![Figure 2: Inference Design](./images/4746e708cf7b61761490f9b2a73cf18528372b11c7862361c52d91135b2d722e.jpg)

**位置**：Section 3.2 Preliminaries
**Caption**：DFlash Inference Design. Hidden context features extracted from the target model are fused and injected into each draft layer's Key-Value cache to enable conditional speculation.

### 客观描述（mimo-v2.5 多模态提取）
- **类型**：流程示意图，展示 DFlash 推测解码的完整架构
- **核心模块**：
  - Target Model（目标模型）→ Target Embedding
  - Draft Layer 1（含 KV Cache、Bidirectional Attention、MLP）→ Draft Layer 2 → ... → Target LM Head（共享）
- **颜色编码**：
  - 蓝色：Fused Target Context Feature（融合目标上下文特征）
  - 黄色：Target Decode Token（目标解码 token）
  - 绿色：Mask Token（掩码 token）
- **数据流**：输入（如 "Diffusion is good"）→ Target Model → 生成目标上下文特征 + 解码 token + mask token → Draft Layer（双向注意力 + MLP）→ Target LM Head → 推测解码输出

### 深度分析

- **独立解读**：展示了 DFlash 推理的完整数据流——目标模型提取特征，通过 KV Cache 持续注入 draft 模型的每一层。关键设计是特征**不经过 Q 投影**，只作为 K/V 条件。

- **对照 caption**：一致。准确描述了"hidden context features → fused → injected into each draft layer's KV cache"的机制。

- **验证的假设**：KV Injection 的核心设计——持续注入 vs 一次性融合。这张图是理解 DFlash 推理的关键。

- **批判性评价**：
  - ✅ 图很清晰，数据流一目了然
  - ✅ 标注了"shared"说明 embedding 和 LM head 是共享的
  - ⚠️ 未展示 KV Injection 的数学细节（公式在 Appendix A.3）
  - ⚠️ 没有展示 draft 模型内部的 masked attention pattern

- **面试价值**：面试时画这张图就能解释清楚 DFlash 的推理流程。关键点：context feature 通过 KV Cache **注入每一层**，而非仅在输入层融合。

---

## Figure 3: Draft Cost Comparison — Draft 延迟对比

![Figure 3: Draft Cost](./images/299a78c361d70f505bbff7dba6ec7e3f8f478b39f5b07943d3333d436d79876a.jpg)

**位置**：Section 3.2 Preliminaries
**Caption**：Draft cost of 1, 3, 5-layer DFlash and 1-layer EAGLE-3.

### 客观描述（mimo-v2.5 多模态提取）
- **类型**：分组柱状图，对比 4 种方法在不同草稿 token 数量下的延迟
- **X 轴**：Number of Draft Tokens（草稿 token 数量，取值 4、8、16）
- **Y 轴**：Latency (ms)，范围 0~20
- **四组柱**：
  - 灰色柱：EAGLE-3
  - 蓝色柱：DFlash (1)（1 层草稿层）
  - 橙色柱：DFlash (3)（3 层草稿层）
  - 绿色柱：DFlash (5)（5 层草稿层）

### 关键数据（mimo-v2.5 + LaTeX 源码交叉验证）

| Draft Tokens | EAGLE-3 (1L) | DFlash (1L) | DFlash (3L) | DFlash (5L) |
|-------------|-------------|-------------|-------------|-------------|
| 4 | ~6ms | ~2ms | ~3ms | ~4ms |
| 8 | ~11ms | ~2ms | ~3ms | ~5ms |
| 16 | ~25ms | ~2ms | ~4ms | ~5ms |

### 深度分析

- **独立解读**：EAGLE-3 的延迟随 token 数**线性增长**（4→7ms, 8→12ms, 16→25ms），而 DFlash 的延迟几乎**恒定**（不随 token 数增长）。即使 5 层 DFlash 生成 16 个 tokens，延迟也只有 ~6ms，远低于 EAGLE-3 生成 16 tokens 的 25ms。

- **对照 caption**：一致。"Draft cost"即 draft 阶段的时间开销。

- **验证的假设**：这是论文最核心的实验支撑——**block diffusion 的并行生成使 draft 延迟与 token 数无关**。这直接解释了为什么 DFlash 可以用更深的模型而不增加延迟。

- **批判性评价**：
  - ✅ 四种配置在相同条件下的公平对比
  - ✅ 数据清晰，趋势明显
  - ⚠️ EAGLE-3 只有 1 层——但论文解释了这是因为自回归 drafting 受延迟限制只能用 1 层（更深会更慢）
  - ⚠️ 缺少 DFlash-8L 的数据（Table 6 消融中有 8 层的结果）

- **面试价值**：这张图是"DFlash 为什么快"的直观答案。串行（线性增长） vs 并行（恒定）——一次前向 pass 生成整个 block。

---

## Figure 4: Training Attention Mask — 训练注意力掩码

![Figure 4: Training Attention](./images/4878d8bbcd0b69acf0dbb498c080796db62b9c0916e8ef78f476ba3be8f7d52d.jpg)

**位置**：Section 4.2 Training
**Caption**：DFlash training attention. The target model provides context features (blue) that condition the draft model. The input consists of clean prompt tokens p and clean response tokens r. Within each masked block, a subset of clean response tokens (yellow) are randomly sampled as anchors, while mask tokens m (green) mark positions for parallel prediction. Invisible tokens (white) denote the attention mask, which enforces causal consistency and prevents inter-block information leakage during training.

### 客观描述（mimo-v2.5 多模态提取）
- **类型**：矩阵结构图，展示目标模型输出和 Mask Blocks 的 token 分布
- **主要组件**：
  - 左侧矩阵：From Target Model（目标模型输出，列含 p1-p4、r1-r2）
  - 右侧矩阵：Mask Blocks（掩码块，列含 r1、\<m> 重复、r2、\<m> 重复、r3、\<m> 重复）
- **颜色编码**：
  - 🔵 蓝色：Target Context Feature（目标上下文特征）
  - 🟢 绿色：Mask Token（掩码 token）
  - 🟡 黄色：Clean Token（干净 token / anchor）
  - ⬜ 白色：Invisible Token（不可见，注意力被屏蔽）
- **结构**：展示了目标模型输出的特征分布和 Mask Blocks 的组织结构

### 深度分析

- **独立解读**：展示了训练时的注意力可见性规则——
  1. 每个 block 内部：双向注意力（mask tokens 可以互相看到）
  2. 每个 block 前面：可以看到所有 prompt 和之前的 clean tokens
  3. 不同 block 之间：互相不可见（防止信息泄漏）
  4. Target context features：所有位置都可见

- **对照 caption**：完全一致。四种颜色的角色描述准确——blue=context feature, yellow=anchor, green=mask, white=invisible。

- **验证的假设**：支持"随机 anchor 采样 + 块间隔离"的训练设计。这确保：
  - 每个 block 独立 denoise（不影响其他 block）
  - Anchor 是 clean token（匹配推理时以 bonus token 为条件的行为）
  - 随机化 anchor 增加训练多样性

- **批判性评价**：
  - ✅ 图复杂但 caption 非常详细，组合起来能完全理解
  - ✅ 颜色编码清晰
  - ⚠️ 初次看可能需要反复对照 caption 才能理解全部细节
  - ⚠️ 没有展示 loss weighting 的可视化（Figure 5 弥补了部分）

- **面试价值**：这张图展示了 DFlash 训练的核心创新——不是标准 block diffusion 的均匀分块，而是随机 anchor + speculative-decoding-aware 的 block 构造。面试时能画出这个 mask 就说明真理解了。

---

## Figure 5: Loss Decay Ablation — 损失衰减消融

![Figure 5: Loss Decay](./images/5547ddba7a0b82aa0c0514e2d329cee3b5f470d0b971edd3d6568c81d20e9c99.jpg)

**位置**：Section A.5.1 Appendix
**Caption**：The loss decay makes training converge faster and better.

### 客观描述（mimo-v2.5 多模态提取）
- **类型**：折线图，对比有/无损失衰减对 Acceptance Length 的影响
- **X 轴**：Epoch（训练轮数，范围 1~9）
- **Y 轴**：Acceptance Length (Math500)，范围 4.2~6.5
- **两条线**：
  - 蓝色线：With loss decay（有损失衰减）
  - 橙色线：Without loss decay（无损失衰减）

### 关键数据

| Epoch | With loss decay | Without loss decay |
|-------|-----------------|--------------------|
| 1 | 4.4 | 4.2 |
| 2 | 5.3 | 5.2 |
| 3 | 5.8 | 5.6 |
| 4 | 6.1 | 6.0 |
| 5 | 6.2 | 6.2 |
| 8-9 | 6.4 | 6.4 |

### 深度分析

- **独立解读**：有 loss decay 的训练在早期 epoch 收敛更快（epoch 3-4 差距最大 ~0.2），最终（epoch 8-9）两者趋于相同。

- **对照 caption**：一致但略有夸大——"faster"确实成立（早期差距明显），但"better"不够convincing（最终值几乎相同）。

- **验证的假设**：位置加权损失（早期 token 权重更大）能加速收敛。这与论文的直觉一致——在投机解码中，早期 token 更重要（一个错误导致后续全废）。

- **批判性评价**：
  - ✅ 消融实验设计干净（只改变 loss weighting，其他不变）
  - ⚠️ 最终 acceptance length 几乎相同（6.4 vs 6.4），说明 loss decay 主要影响收敛速度而非最终性能
  - ⚠️ 只在一种配置下测试，不同 block size / draft 层数下是否一致？

- **面试价值**：证明了"投机解码中早期 token 更重要"这个直觉在训练中确实有效。是一个简单但有用的 trick。

---

## Table 1: 主实验 — Qwen3 系列 Speedup（Greedy + Sampling）

**位置**：Section 5.1

### 关键数据摘要

| 模型 | 方法 | Greedy 平均加速 | Sampling 平均加速 |
|------|------|----------------|------------------|
| Qwen3-4B | DFlash (BS=16) | 4.0× | 3.5× |
| Qwen3-8B | DFlash (BS=16) | 4.9× | 4.1× |
| Qwen3-8B | EAGLE-3 (tree=16) | 2.1× | 1.9× |
| Qwen3-8B | EAGLE-3 (tree=60) | 2.5× | 2.3× |

### 分析
- DFlash 在 greedy 和 sampling 两种模式下都显著优于 EAGLE-3
- 即使对比 EAGLE-3 的最大 tree size (60)，DFlash 仍然快约 2×
- 接受长度 τ：DFlash (16) ≈ 6.5，EAGLE-3 (16) ≈ 3.5，几乎翻倍
- **面试要点**：DFlash 的优势来源于高接受长度（KV Injection）+ 低 draft 延迟（并行生成），两者乘法叠加

---

## Table 2: Thinking Mode（推理模式）

**位置**：Section 5.2

### 分析
- 开启 thinking mode 后 DFlash 达到 4.5×（Qwen3-4B）和 3.9×（Qwen3-8B）加速
- Thinking mode 生成长度更长，投机解码的加速效果更显著
- **关键洞察**：对 CoT 推理模型（如 o1、DeepSeek-R1），DFlash 的价值更大——因为推理时间占主导

---

## Table 3: SGLang 服务框架（生产环境）

**位置**：Section 5.3

### 关键数据（Qwen3-8B, B200, FA4 backend）

| Concurrency | Speedup |
|------------|---------|
| 1 | 5.1× |
| 8 | 3.5× |
| 16 | 2.8× |
| 32 | 2.8× |

### 分析
- 低并发时加速最大（5.1×），高并发时下降到 2.8×
- 原因：高并发时验证阶段的 compute 变成瓶颈（GPU compute-bound 而非 memory-bound）
- **面试要点**：投机解码在低并发/低 batch size 时效果最好，因为此时 GPU 利用率最低（memory-bound），draft+verify 能更好地利用闲置算力

---

## Table 4: 长上下文适应

**位置**：Section 5.4

### 分析
- 基础模型（4K context 训练）在 16K context 时接受长度从 ~7.7 降到 ~5.5
- 仅用 **1.6K 长上下文样本 fine-tune 3 个 epoch**，即可恢复甚至超过表现
- 说明 KV Injection 中的 target context feature 在长上下文下仍然有效
- **面试要点**：DFlash 的 domain 适应成本极低（少量样本 + 几个 epoch）

---

## Table 5: LLaMA-3.1-8B 在 SGLang + B200 上的结果

**位置**：Section 5.5.1

### 分析
- 在非 Qwen 模型上也有效，证明方法的通用性
- DFlash (BS=10) 在所有任务和并发级别上都优于 EAGLE-3
- **面试要点**：DFlash 不依赖特定模型架构，适用于任何 autoregressive LLM

---

## Table 6: Draft 层数消融

**位置**：Section 5.5.2

### 关键数据

| Draft 层数 | 平均 τ | 平均 Speedup |
|-----------|--------|-------------|
| 1 层 | 3.85 | 3.01× |
| 3 层 | 5.47 | 4.07× |
| 5 层 | 6.49 | **4.42×** |
| 8 层 | 7.16 | 4.18× |

### 分析
- 接受长度随层数单调增加（3.85→7.16）
- 但 8 层的加速反而低于 5 层——因为 draft 延迟增加了
- **5 层是最佳平衡点**
- **面试要点**：更深 = 更准但更慢，最优层数取决于具体部署场景。这是 Figure 3 所展示的"恒定延迟"的一个 nuance——延迟不是完全恒定，只是不随 token 数增长，但随层数增长

---

## Table 7: 目标特征层数消融

**位置**：Section 5.5.3

### 分析
- 5 层 target features > 3 层（接受长度更高）
- 代价：训练时存储 hidden states 线性增长
- **面试要点**：更多的目标特征 = 更丰富的上下文 = 更高的接受率。但训练成本增加

---

## Table 8: 训练/推理 Block Size 不匹配消融

**位置**：Section 5.5.4

### 关键发现

| 训练 BS | 推理 BS | Speedup |
|--------|--------|---------|
| 16 | 8 | 好 |
| 16 | 16 | 最好 |
| 8 | 8 | 好 |
| 8 | 16 | 差 |

### 分析
- 大 BS 训练 → 小 BS 推理：泛化好（模型见过更长的依赖）
- 小 BS 训练 → 大 BS 推理：泛化差（模型没学过这么长的依赖）
- **面试要点**：实际部署时建议用较大的 block size 训练，推理时可以灵活调小

---

## Table 9: KV Injection vs Input Fusion 消融

**位置**：Section 5.5.5

### 分析
- 对比了两种条件注入方式（Input Fusion vs KV Injection）× 两种 drafting 方式（AR vs Block Diffusion）
- **结论**：KV Injection 在两种 drafting 方式下都优于 Input Fusion
- Block Diffusion + KV Injection = 最优组合
- **面试要点**：这证明了 DFlash 的两个创新是互补的，不是单独有效的

---

## Table 10: 无目标特征的 Diffusion Draft（反面案例）

**位置**：Appendix A.3

### 分析
- 5 层 block diffusion draft 模型，**不使用**目标模型的 context feature
- 加速仅 2-3×，远低于有 context feature 的 5-6×
- **证明了核心假设**：没有目标模型的内部表征，diffusion draft 必须从零预测，质量很差

---

## Table 11: 更多模型在 SGLang 上的结果

**位置**：Appendix A.4

### 分析
- 覆盖 Qwen3.5-9B/27B/35B-A3B、Qwen3-Coder-Next、GPT-OSS-20B/120B
- DFlash 在所有模型上都优于 native MTP（当两者都可用时）
- GPT-OSS-120B（最大模型）加速 1.3-1.7×，说明超大模型上 draft 质量更难保证
- **面试要点**：DFlash 的有效性不限于 Qwen 系列，具有较好的通用性

---

## Table 12: vLLM 结果

**位置**：Appendix A.4

### 分析
- 在另一个主流推理框架 vLLM 上也有效
- Qwen3.5-9B：低并发 4.0-4.6×，高并发 1.9-2.1×
- **面试要点**：DFlash 不依赖特定推理框架，已集成到 SGLang 和 vLLM

---

## Table 13: 随机 Anchor 采样消融

**位置**：Appendix A.5.2

### 关键数据

| 设置 | Math500 Speedup | Math500 τ | HumanEval Speedup | HumanEval τ |
|------|----------------|-----------|------------------|-------------|
| Standard（固定分块） | 4.13× | 4.94 | 3.29× | 3.86 |
| Sample（随机 anchor） | 4.69× | 5.64 | 3.90× | 4.61 |

### 分析
- 随机 anchor 在所有任务上都优于固定分块
- 接受长度提升 ~14%（4.94→5.64），加速提升 ~14%（4.13→4.69）
- **面试要点**：随机 anchor 是一种 data augmentation——让模型在训练时看到更多样的上下文，提高泛化性
