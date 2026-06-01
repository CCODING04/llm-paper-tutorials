# 📖 GPT-3: Language Models are Few-Shot Learners

> **论文**：Brown et al., 2020 (OpenAI) | NeurIPS 2020
>
> **一句话总结**：175B 参数的自回归语言模型，通过 in-context learning（few-shot prompt）在不做任何梯度更新的情况下，在多种 NLP 任务上接近甚至超越 fine-tuned 模型。

---

# 第一层：鸟瞰

## 🎯 核心贡献

1. **In-Context Learning**：首次系统定义和评估 zero-shot / one-shot / few-shot 三种范式——不做梯度更新，只通过 prompt 中的示例让模型适配任务
2. **175B 参数模型**：当时最大的语言模型，8 种规模横跨 3 个数量级（125M-175B），系统验证 scaling
3. **Few-shot 接近 SOTA**：在 SuperGLUE、TriviaQA、翻译等任务上，few-shot 结果接近甚至超越了 fine-tuned 模型
4. **Meta-learning 视角**：把 in-context learning 理解为"预训练中学会的元学习能力"——模型规模越大，元学习能力越强

## 📍 知识网络定位

```
GPT-2 (2019.02) → zero-shot prompt（1.5B，效果有限）
         ↓
   【GPT-3 (2020.06)】→ few-shot prompt（175B，接近 fine-tuned）
         ↓
   Scaling Laws (2020.09) → 系统化 scaling 观察
   Chinchilla (2022.03) → 质疑数据量不够
   InstructGPT (2022.01) → GPT-3 + RLHF 对齐
   ChatGPT (2022.11) → 产品化
```

**一句话给面试官**：GPT-3 证明了两件事——(1) 175B 模型通过 few-shot prompt 可以接近 fine-tuned 效果；(2) 更大模型不只是"更强"，而是"更善于 in-context learning"（meta-learning 能力随规模涌现）。

---

# 第二层：精读

## 1. 引言：为什么要做 GPT-3？

### 三个问题

引言提出微调范式的三个根本问题：

**问题 1：实用性**
> "There exists a very wide range of possible useful language tasks...For many of these tasks it is difficult to collect a large supervised training dataset."

每个新任务都需要大量标注数据——这限制了 NLP 的应用范围。

**问题 2：泛化性**
> "The potential to exploit spurious correlations in training data fundamentally grows with the expressiveness of the model and the narrowness of the training distribution."

微调数据太窄 → 模型学到了虚假关联 → IID 评估虚高 → 分布外表现差。

> ❓ **这个论点后来被验证了吗？** 是的。RLHF 模型（ChatGPT）在某些 benchmark 上表现极好但在分布外仍然脆弱。GPT-3 指出的"虚假关联"问题远未解决。

**问题 3：人类类比**
> "Humans do not require large supervised datasets to learn most language tasks – a brief directive in natural language or at most a tiny number of demonstrations is often sufficient."

人类只需要一句话指令或一两个示例就能学会新任务。我们的 NLP 系统应该也能。

### 从 GPT-2 到 GPT-3 的逻辑跳跃

GPT-2 的论点：语言模型能做 zero-shot 多任务学习。
GPT-2 的结果：zero-shot 效果有限（CoQA 55 F1，翻译 5 BLEU）。

GPT-3 的假设：也许不是思路错了，而是**模型不够大**。

> "Since in-context learning involves absorbing many skills and tasks within the parameters of the model, it is plausible that in-context learning abilities might show similarly strong gains with scale."

这个假设被 Figure 1.2 验证了——175B 的 few-shot 曲线斜率远大于 1.3B。

### 四种任务适配方式的定义

GPT-3 论文最重要的概念贡献是明确定义了 zero/one/few-shot：

| 方式 | 给模型什么 | 梯度更新 | 类比 |
|------|----------|---------|------|
| **Zero-shot** | 任务描述 | ❌ | "请翻译成法语" |
| **One-shot** | 任务描述 + 1 个示例 | ❌ | 给工人看一个例子 |
| **Few-shot** | 任务描述 + K 个示例 | ❌ | 给工人看几个例子 |
| Fine-tuning | 大量标注数据 | ✅ | 传统 ML |

> 💡 **关键区分**：前三种方式**不做任何梯度更新**——所有"学习"都发生在 forward pass 中。这是 in-context learning 和 fine-tuning 的本质区别。

## 2. 方法

### 2.1 模型架构

GPT-3 基于 GPT-2，但有两个改动：

**改动 1：交替的稠密和稀疏注意力**

论文说 "we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer"。

直觉：不是每层都做全注意力（$O(n^2)$），而是交替使用：
- 稠密注意力：所有 token 互相 attend
- 稀疏注意力：只 attend 局部窗口 + 少量全局 token

这降低了长序列的计算成本，同时保持了对长距离依赖的建模能力。

> ❓ **但论文没有深入讨论这个改动的消融**——稀疏注意力对性能的影响有多大？后来 LLaMA 等模型用不同的稀疏模式（如 Sliding Window Attention），说明注意力稀疏化确实有用。

**改动 2：上下文长度翻倍（1024 → 2048）**

Few-shot 需要把示例放在上下文里——上下文越长，能放的示例越多。2048 token 允许放 ~10-100 个示例（取决于任务）。

### 2.2 训练数据

| 数据源 | 比例 | 权重（训练中采样概率） | 大小 |
|--------|------|---------------------|------|
| Filtered Common Crawl | 60% | 1.0x | 410B tokens |
| WebText2 | 19% | 3.4x | 19B tokens |
| Books1 | 7% | 1.9x | 12B tokens |
| Books2 | 7% | 1.9x | 55B tokens |
| Wikipedia | 3% | 3.4x | 10B tokens |

**关键设计**：
1. **权重 > 1 意味着过采样**：WebText2 和 Wikipedia 被过采样 3.4 倍——高质量数据被多次训练
2. **Common Crawl 过滤**：从 ~45TB 原始数据过滤到 ~570GB——去重、质量过滤、去除测试集
3. **总训练量**：300B tokens（约等于 3.3 epoch of the weighted dataset）

> ❓ **为什么只训练 300B tokens？** 后来 Chinchilla 证明 175B 参数的模型应该用 ~3.7T tokens 训练（约 12x 更多）。GPT-3 确实 undertrained。

> ❓ **数据污染问题**：论文花了整篇 Section 4 讨论数据污染（训练数据中包含测试集内容）。这是大模型评估的核心问题——后来几乎所有大模型论文都要分析这个问题。

### 2.3 训练过程

| 项目 | 值 |
|------|-----|
| 硬件 | V100 GPU 集群 |
| 训练时间 | ~数周（论文未公开确切数字） |
| 估算成本 | ~$4.6M（外界估算） |
| 总计算量 | ~3.14 × 10²³ FLOPS |

> 💡 **为什么公开 8 种规模的模型？** 为了验证 scaling laws——用 8 个数据点画出性能随规模的曲线，证明增长是平滑的幂律关系。

## 3. 实验：核心结果

### 3.1 语言建模

GPT-3 175B 在 Penn Tree Bank 上达到 **20.50 PPL**（zero-shot SOTA），LAMBADA 上达到 **76.2%** 准确率。

### 3.2 问答（Closed-book QA）

| 任务 | GPT-3 Few-shot | 之前 SOTA (fine-tuned) |
|------|---------------|---------------------|
| TriviaQA | **71.2%** | 68.0% |
| Natural Questions | 29.9% | 36.6% |

> 💡 TriviaQA 超越了 fine-tuned 模型！这是 GPT-3 最令人印象深刻的结果之一——一个不做微调的模型在知识问答上超越了专门训练的模型。说明 175B 的参数里存储了大量知识。

### 3.3 翻译

| 方向 | GPT-3 Few-shot | 之前无监督 SOTA | 之前有监督 SOTA |
|------|---------------|---------------|---------------|
| En→Fr | 29.7 BLEU | 33.5 BLEU | 40+ BLEU |
| Fr→En | 36.2 BLEU | 33.5 BLEU | 40+ BLEU |

> 对比 GPT-2（En→Fr 5 BLEU），GPT-3 达到 29.7 BLEU——**提升了 6 倍**！
>
> 为什么？两个原因：1) 模型大了 117 倍；2) 训练数据中包含了更多法语（Common Crawl 中有大量多语言内容）。

### 3.4 SuperGLUE

SuperGLUE 是"通用语言理解"的终极测试（比 GLUE 更难）：

| 任务 | GPT-3 Few-shot | SOTA (fine-tuned) | 人类 |
|------|---------------|------------------|------|
| BoolQ | 76.4% | 83.0% | 88% |
| COPA | 92.0% | 94.0% | 100% |
| WSC | 80.1% | 91.3% | 100% |
| MultiRC | 75.9% | 84.0% | 98% |

> 💡 COPA 92%——几乎接近人类。但 WSC（Winograd Schema）只到 80%，说明常识推理仍然是弱点。

### 3.5 算术（新能力涌现）

GPT-3 在两位数加法上达到 ~100% 准确率，三位数 ~80%，四位数 ~25%，五位 ~9%。

> ❓ **为什么算术重要？** 因为训练数据中不可能包含所有加法组合——模型必须**学会加法算法**而非死记硬背。这暗示了大模型的某种"推理"能力。但实际上 GPT-3 的算术能力可能主要是模式匹配——后来发现只要数字位数稍大，准确率就暴跌。

### 3.6 合成任务（新词学习）

论文设计了"学习新词"任务：先给一个新词的定义，然后要求模型用这个词造句。

> 💡 这是 in-context learning 最直观的演示——模型在 forward pass 中"学会"了一个它从未见过的词，并正确使用。这比算术更能说明 meta-learning 能力。

### 3.7 规模效应的核心发现

**发现 1：Few-shot 增长比 Zero-shot 更快**（Figure 1.3）

从 0.1B 到 175B：
- Zero-shot: 25 → 42（+17）
- Few-shot: 30 → 58（+28）

**Few-shot 的 scaling 坡度更陡**——大模型不仅是"更强"，而是"更善于利用上下文示例"。

**发现 2：有些能力只在特定规模涌现**

算术、新词学习、模式操纵等任务——小模型几乎完全不会，只有达到一定规模后才开始"涌现"。

> ❓ **"涌现"是真的还是评估的假象？** 后来有研究（Schaeffer et al., 2023）质疑"涌现"可能只是评估指标的假象——如果你把准确率换成对数几率，增长就变平滑了。但即使如此，实际使用中"小模型完全不会，大模型突然能用"的体验是真实的。

---

# 第三层：批判性思考

## 🤔 设计决策分析

### 为什么不做 fine-tuning？

论文的立场：要证明 in-context learning 够用。

> ❓ **批判**：论文最后承认 "fine-tuning is a promising direction for future work"。后来 InstructGPT（GPT-3 + RLHF fine-tuning）确实大幅提升了效果。不做 fine-tuning 的"纯粹"路线在 ChatGPT 时代被放弃了——**最终实用的系统还是需要某种形式的对齐训练**。

### 为什么用自回归 LM 而不是 MLM？

GPT-3 继续用自回归（单向）而非 BERT 的 MLM（双向）。

> **论文隐含的理由**：自回归 LM 天然支持文本生成，而 few-shot 需要"生成"答案。MLM 的填空任务无法自然地做条件生成。

> ❓ **但后来的 T5/Flan-T5 证明编码器-解码器架构也能做 few-shot**。自回归不是唯一选择，但确实是最统一的选择——所有任务都化为"文本续写"。

### 175B 是必要的吗？

**论文的核心论点**：是的，因为 few-shot 能力需要足够大的模型才能涌现。

> ❓ **批判**：后来 LLaMA-2 7B 在 fine-tuning 后效果接近 GPT-3 175B few-shot——说明**小模型 + fine-tuning** 可能比**大模型 + few-shot** 更高效。但 GPT-3 的贡献是证明了"不微调也能做"的可能性，这个方向后来被 RLHF 延续。

## ⚠️ 局限性

### 论文自己承认的（Section 5）

1. **文本生成弱点**：在长文本生成中经常"跑题"或重复
2. **常识推理不足**：物理/社交常识仍然薄弱
3. **NLI（自然语言推理）弱**：ANLI、RACE 上表现差
4. **无法学习新的事实**：模型的知识冻结在训练数据中
5. **预训练数据的偏见**：性别、种族、宗教偏见

### 自己发现的

1. **Undertrained**：只训练了 300B tokens——后来 Chinchilla 证明应该训练 ~3.7T
2. **数据污染**：虽然分析了 8-gram 重叠，但无法完全消除
3. **评估不严格**：很多任务的 prompt 设计对结果影响很大，论文没有系统分析 prompt 敏感性
4. **训练成本**：~$4.6M 的估算成本——只有极少数组织能负担
5. **Few-shot 示例数量有限**：上下文 2048 token 只能放 ~10-100 个示例——远远不够"学会"复杂任务

## 🎯 面试视角

**Q1: GPT-3 的核心创新是什么？和 GPT-2 的区别？**

> **标准回答**：核心创新是系统定义和评估了 in-context learning（zero/one/few-shot），并证明 few-shot 在 175B 规模下接近 fine-tuned 效果。GPT-2 只做了 zero-shot 且效果有限。
>
> **追问：为什么 175B 行但 1.5B 不行？** 因为 in-context learning 是一种 meta-learning 能力——需要在预训练中"学会如何从上下文中学习"。这种能力需要足够的模型容量才能涌现。

**Q2: In-context learning 的原理是什么？**

> **标准回答**：有两种理解方式。论文的视角是 meta-learning——预训练数据中包含大量隐式的"任务"模式，模型在训练中学会了"识别任务并适配"。另一种视角是"条件生成"——few-shot 示例作为条件，引导模型生成符合任务要求的输出。
>
> **追问：哪种更准确？** 目前学术界没有定论。后来的工作（Akyürek et al., 2022）发现 Transformer 的 in-context learning 实际上在隐式执行梯度下降——这更支持 meta-learning 的视角。

**Q3: GPT-3 的 scaling laws 表现如何？**

> **标准回答**：8 种规模的模型验证了性能随规模平滑增长（幂律关系）。关键发现：few-shot 的增长比 zero-shot 更快——大模型是"更好的 meta-learner"。但所有模型都只训练了 300B tokens，后来 Chinchilla 证明这不够。

**Q4: GPT-3 的弱点是什么？**

> **标准回答**：NLI（自然语言推理）弱（ANLI 上 barely above random）、长文本生成容易跑题、无法学习新事实、训练数据偏见。论文 Section 5 非常诚实地列出了这些局限。
>
> **追问：哪些后来被解决了？** NLI 通过 RLHF 大幅改善；长文本生成通过更好的 sampling 策略和 RLHF 改善；但"学习新事实"的问题（幻觉/hallucination）至今未完全解决。

**Q5: 为什么 GPT-3 不做 fine-tuning？**

> **标准回答**：论文要研究"不微调能走多远"。这是方法论选择而非技术限制。后来 InstructGPT 证明 fine-tuning（RLHF）在 GPT-3 基础上大幅提升效果——说明最终路线是"预训练 + 对齐训练"。

---

# 第四层：知识网络

## 📅 时间线

```
GPT-2 (2019.02) → zero-shot prompt（1.5B，效果有限）
T5 (2019.10) → 编码器-解码器 + text-to-text 统一范式
    【GPT-3 (2020.06)】→ few-shot prompt（175B，接近 fine-tuned）
Scaling Laws (2020.09) → 系统化 scaling（Kaplan et al.）
Chinchilla (2022.03) → 计算最优需要更多数据
PaLM (2022.04) → 540B，验证 scaling 继续
InstructGPT (2022.01) → GPT-3 + RLHF
ChatGPT (2022.11) → 产品化
Flan-T5 (2022.10) → 指令微调也能实现 few-shot 效果
```

## ↔️ 同期/后续对比

| 维度 | GPT-3 (2020.06) | T5 (2019.10) | Chinchilla (2022.03) |
|------|----------------|-------------|---------------------|
| 参数量 | 175B | 11B | 70B |
| 训练 tokens | 300B | 1T | 1.4T |
| 下游方式 | Few-shot prompt | Fine-tuning | Fine-tuning |
| 核心论点 | 不微调也行 | 统一 text-to-text | 数据量比模型大小更重要 |

## 🔗 知识关联

- **llm-math-foundations Ch02**：GPT-3 的训练目标 = 最大似然估计
- **llm-math-foundations Ch03**：幂律 scaling 关系（Figure 1.3 验证）
- **本系列 03-GPT-2**：GPT-3 的直接前身——从 zero-shot 到 few-shot
- **本系列 05-Chinchilla**：质疑 GPT-3 的数据量不够
- **本系列 06-InstructGPT**：GPT-3 + RLHF 的对齐方案

## 📊 GPT-3 的遗产

| 创新 | 被继承 | 被抛弃/修正 |
|------|--------|-----------|
| In-context learning | 所有后续大模型 | Few-shot 被 instruction following 取代 |
| 175B 规模 | 后来万亿参数 | 证明"足够大就行" |
| 不做 fine-tuning | GPT-3 API | 被 RLHF (InstructGPT) 修正 |
| 数据混合策略 | 所有后续模型 | Common Crawl → 更好的过滤 |
| 数据污染分析 | 所有后续模型 | 成为标准流程 |

---

## ❓ 深度思考题

1. **概念题**：GPT-3 说 in-context learning 是 meta-learning。但 meta-learning 通常有"内外循环"——GPT-3 的内外循环分别是什么？如果预训练是"外循环"，那"内循环"发生在哪里？

2. **设计题**：如果你来设计 GPT-3 的数据混合策略，你会怎么调整权重？为什么 Wikipedia 和 WebText2 被过采样 3.4 倍而 Common Crawl 只有 1.0 倍？

3. **批判题**：GPT-3 在算术任务上的"涌现"能力是真正的推理还是模式匹配？设计一个实验来区分这两种解释。

4. **面试题**：面试官问"为什么 ChatGPT 能对话但 GPT-3 不能？"你怎么从技术角度解释从 GPT-3 到 ChatGPT 的演进？

5. **拓展题**：GPT-3 的 few-shot 能力在 175B 时"涌现"。如果用 13B 模型但给 10 倍多的 few-shot 示例（假设上下文无限长），能追上 175B 吗？这说明了什么？

6. **实现题**：GPT-3 用了"交替稠密和稀疏注意力"。如果你要实现一个简单的稀疏注意力（只 attend 局部窗口 + 全局 token），你会怎么设计？复杂度是多少？

7. **哲学题**：GPT-3 在 few-shot 下"学会"了一个从未见过的新词——这是"学习"还是"条件生成"？如果一个系统在 forward pass 中就"学会了"，它的"学习"和传统意义上的学习有什么本质区别？

8. **对比题**：GPT-3 证明了大模型 + few-shot 接近 fine-tuned 效果。但 LLaMA-2 7B + fine-tuning 效果也接近 GPT-3 175B few-shot。哪种路线更"正确"？从效率、公平性、可控性三个角度分析。

---

## 📚 延伸阅读

| 论文 | 年份 | 和 GPT-3 的关系 |
|------|------|----------------|
| **Scaling Laws** (Kaplan et al.) | 2020 | GPT-3 验证了 scaling laws 的预测 |
| **Chinchilla** (Hoffmann et al.) | 2022 | 质疑 GPT-3 undertrained——需要 12x 更多数据 |
| **InstructGPT** (Ouyang et al.) | 2022 | GPT-3 + RLHF——解决对齐问题 |
| **PaLM** (Chowdhery et al.) | 2022 | 540B，验证 scaling 继续有效 |
| **Flan-T5** (Chung et al.) | 2022 | 证明指令微调也能实现 few-shot 效果 |
| **Emergent Abilities** (Wei et al.) | 2022 | 系统研究大模型的涌现能力 |
| **GPT-2** (Radford et al.) | 2019 | GPT-3 的前身 |
