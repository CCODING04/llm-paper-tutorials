# 📖 InstructGPT: Training language models to follow instructions with human feedback

> **论文**：Ouyang et al., 2022 (OpenAI) | NeurIPS 2022
>
> **一句话总结**：用 RLHF（人类反馈强化学习）微调 GPT-3，使 1.3B 模型输出被偏好度超过 175B GPT-3——"对齐比规模更重要"。

---

# 第一层：鸟瞰

## 🎯 核心贡献

1. **RLHF 工程化落地**：首次将 RLHF 从研究原型（Summarize）扩展到通用指令跟随——SFT → RM → PPO 三阶段流水线
2. **对齐 > 规模**：1.3B InstructGPT 输出被偏好度超过 175B GPT-3（100x 参数差距！）——证明对齐比堆参数更有效
3. **Alignment Tax 概念**：对齐有代价——RLHF 会损害某些 NLP 能力（SQuAD/DROP/WMT 下降）。PPO-ptx 通过混合预训练更新缓解
4. **Truthfulness 提升**：TruthfulQA 上 truthful+informative 的比例翻倍；幻觉率从 41% 降到 21%

## 📍 知识网络定位

```
GPT-3 (2020) → 175B 但不follow指令（"misaligned"）
Christiano et al. (2017) → RLHF 原始方法
Stiennon et al. (2020) → RLHF 用于摘要
         ↓
   【InstructGPT (2022.03)】→ SFT+RM+PPO 三阶段 RLHF
         ↓
   ChatGPT (2022.11) → InstructGPT + 对话优化 → 爆发
   LLaMA-2 Chat (2023.07) → 开源 RLHF
   Constitutional AI (2022.12) → AI 自我对齐（减少人类标注）
```

---

# 第二层：精读

## 1. 引言：为什么需要 RLHF？

### 核心问题

> "The language modeling objective—predicting the next token on a webpage—is different from 'follow the user's instructions helpfully and safely'."

**目标错位**：语言建模 ≠ 指令跟随。大模型擅长预测下一个 token，但不擅长：
- 遵循用户指令
- 说真话（而非编造）
- 避免有害输出

### 三个对齐目标

1. **Helpful**（有用）：帮助用户完成任务
2. **Honest**（诚实）：不编造信息
3. **Harmless**（无害）：不产生有害内容

> ❓ **这三个目标有冲突吗？** 有。比如用户要求"写一个关于某群体的偏见笑话"——helpful 要求满足，harmless 要求拒绝。InstructGPT 的处理：优先 harmless。

## 2. 方法：三阶段流水线

### Stage 1: SFT（Supervised Fine-Tuning）

- 数据：~13K 条人工撰写的 prompt + 高质量回答
- 来源：OpenAI API 提交的 prompt + 标注员自己写的 prompt
- 训练：在 GPT-3 上做标准监督微调（2-3 epochs）
- 结果：175B SFT 模型已经比 GPT-3 好，但还不够

### Stage 2: RM（Reward Model）

- 数据：标注员对模型多个输出进行排序（A > B > C > D）
- 训练目标：学习人类偏好函数
  $$\text{loss} = -\log \sigma(r(x, y_w) - r(x, y_l))$$
  - $r(x, y)$：reward model 对 prompt x + 回答 y 的打分
  - $y_w$：被偏好的回答
  - $y_l$：不被偏好的回答
- RM 大小：6B（不是 175B——更大反而不稳定）

> ❓ **为什么 RM 用 6B 而不是 175B？** 论文发现更大的 RM 过拟合更严重，且推理成本更高。6B 在验证集上表现最好。这也是后来 ChatGPT 的 RM 选型经验。

### Stage 3: PPO（强化学习微调）

- 环境：语言模型生成文本 → RM 打分 → reward signal
- 算法：PPO（Proximal Policy Optimization）
- 目标函数：
  $$\text{objective} = \mathbb{E}[r(x,y)] - \beta \cdot KL(\pi_\theta || \pi_{\text{ref}})$$
  - 第一项：最大化 RM 奖励
  - 第二项：KL 散度惩罚，防止偏离 SFT 模型太远
- **PPO-ptx**：混合预训练更新（部分 update 来自原始 GPT-3 的语言建模目标）

> ❓ **为什么需要 KL 惩罚？** 如果只优化 RM reward，模型可能"reward hacking"——生成 RM 给高分但实际无意义的文本。KL 惩罚确保输出不会偏离太远。

> ❓ **PPO-ptx 是什么？** InstructGPT 的关键改进。纯 PPO 会导致 "alignment tax"（SQuAD/DROP 等能力下降）。PPO-ptx 在 PPO 更新中混合了预训练数据的更新——保持通用 NLP 能力。代价是稍微降低对齐效果，但整体更好。

### 数据细节

| 数据集 | 数量 | 用途 |
|--------|------|------|
| SFT 训练集 | ~13K prompts + 演示 | Stage 1 |
| RM 训练集 | ~33K prompts + 排序 | Stage 2 |
| RLHF/PPO | ~31K prompts | Stage 3 |

**标注员**：40 人团队，通过筛选测试录用。

### API Prompt 分布

| 用途类型 | 比例 |
|---------|------|
| Generation | 45.6% |
| Open QA | 12.4% |
| Brainstorming | 11.2% |
| Chat | 8.4% |
| Rewrite | 6.6% |
| Summarization | 4.2% |
| Classification | 3.5% |

> 💡 **面试价值**：InstructGPT 的训练数据来自真实的 API 使用场景——不是学术 NLP 任务。这是它比 FLAN/T0 更好的原因。

## 3. 实验结果

### 3.1 人类偏好评估（核心结果）

**Figure 1**：人类评估各模型输出被偏好度（vs 175B SFT baseline）

| 模型 | Winrate vs 175B SFT |
|------|-------------------|
| GPT-3 (175B) | ~10% |
| GPT-3 + few-shot prompt | ~20% |
| **InstructGPT 1.3B** | **~60%** |
| InstructGPT 6B | ~70% |
| InstructGPT 175B | **~85%** |

> 💡 **最震撼的结果**：1.3B InstructGPT 的输出被偏好度远超 175B GPT-3——尽管参数量只有 1/134。

> ❓ **为什么偏好度差距这么大？** 因为 GPT-3 不 follow 指令——它倾向于"续写"而非"回答"。比如用户问"解释量子力学"，GPT-3 可能续写"解释量子力学的难点在于..."而非真正解释。InstructGPT 会直接给出解释。

### 3.2 TruthfulQA

- InstructGPT truthful+informative 比例约 **GPT-3 的 2 倍**
- 在非对抗性问题上同样有效
- 闭域任务幻觉率：**21% vs 41%**（InstructGPT vs GPT-3）

### 3.3 Alignment Tax

纯 PPO 在某些 NLP 基准上退化：

| 数据集 | GPT-3 | SFT | PPO | PPO-ptx |
|--------|-------|-----|-----|---------|
| SQuAD | 89 | 86 | 79 | **84** |
| DROP | 76 | 73 | 67 | **73** |
| WMT fr→en | 33 | 31 | 25 | **32** |

> ❓ **为什么 RLHF 会损害 NLP 能力？** 因为 RM 是基于人类偏好训练的——人类偏好不等于 NLP 基准测试的偏好。RM 可能学到了"简洁、礼貌"比"准确、完整"更重要。PPO-ptx 通过混合预训练更新缓解了这个问题。

### 3.4 vs FLAN/T0

| 模型 | Winrate vs baseline |
|------|-------------------|
| FLAN (GPT-3) | 29.8% |
| T0++ | 26.8% |
| **InstructGPT** | **73.4%** |

> 💡 **关键洞察**：在真实 API 分布上，FLAN/T0 只比 GPT-3 稍好，远不如 InstructGPT。因为 FLAN/T0 用的是学术 NLP 任务数据，和真实用户需求分布不匹配。

---

# 第三层：批判性思考

## 🤔 设计决策分析

### 标注员只用了 40 人

- **问题**：40 人的偏好能代表"人类价值观"吗？
- **论文承认**："This procedure aligns the behavior of GPT-3 to the stated preferences of a specific group of people (mostly our labelers and researchers), rather than any broader notion of 'human values'"
- **影响**：标注员的偏好有偏差（文化、教育、语言）。后来 Constitutional AI 用 AI 辅助减少人类标注偏差。

### RM 的 scaling 问题

- RM 用 6B，但后来发现更大的 RM 可以工作（只要数据量足够）
- Reward hacking：RM 可能有系统性偏差——模型可能学到"讨好 RM"而非"真正对齐"
- **后来的改进**：迭代 RLHF（多次训练 RM + PPO）、DPO（直接偏好优化，跳过 RM）

### 对齐的"毒性悖论"

论文承认：当被指示"尽可能有偏见"时，InstructGPT 比 GPT-3 生成更毒性的输出——因为它更擅长 follow 指令。

> ❓ **这揭示了什么？** "对齐"是双刃剑——follow 指令的能力越强，被恶意使用的风险越大。后来的解决方案：system prompt + 安全过滤 + 拒绝回答。

### 数据分布偏差

API prompt 分布（Generation 45.6%, Open QA 12.4%）和学术 NLP 任务分布完全不同。InstructGPT 在学术 NLP 上退化是可以预期的。

## ⚠️ 局限性

1. **Still makes simple mistakes**：编造事实、过度规避、无法检测错误前提
2. **只对齐了 40 人**的偏好——不代表全人类
3. **只做了英文**——多语言能力未验证
4. **RM 过拟合风险**：小数据 + 大模型 = 过拟合
5. **Alignment Tax**：通用 NLP 能力下降，PPO-ptx 只能缓解不能消除

## 🎯 面试视角

**Q1: InstructGPT 的三阶段流程？**

> A: (1) SFT：用人工撰写的 prompt+回答做监督微调；(2) RM：训练 reward model 学习人类偏好（排序 → 打分函数）；(3) PPO：用 RM 作为 reward，PPO 算法微调 SFT 模型。关键改进：PPO-ptx 混合预训练更新缓解 alignment tax。

**Q2: 为什么 1.3B InstructGPT 能超过 175B GPT-3？**

> A: 因为 GPT-3 不 follow 指令——它做的是"续写"而非"回答"。InstructGPT 通过 RLHF 学会了"根据指令生成有用的回答"。这是目标对齐的力量：正确的目标 > 更大的模型。

**Q3: 什么是 Alignment Tax？怎么缓解？**

> A: 对齐有代价——RLHF 会损害某些 NLP 能力（如 SQuAD 下降 10 分）。原因是 RM 偏好和 NLP 基准偏好不一致。PPO-ptx 通过混合预训练数据的更新来缓解——代价是稍微降低对齐效果。

**Q4: RLHF 的 reward hacking 问题？**

> A: 模型可能学到"讨好 RM"而非真正对齐——比如生成 RM 给高分但实际无意义的文本。缓解方法：KL 惩罚（防止偏离太远）、迭代 RLHF（多次更新 RM）、DPO（直接优化偏好，跳过 RM）。

**Q5: InstructGPT 和 ChatGPT 的关系？**

> A: ChatGPT 是 InstructGPT 的对话优化版——同样的 RLHF 三阶段，但训练数据更偏对话场景。InstructGPT 是 ChatGPT 的技术基础和论文版本。

---

# 第四层：知识网络

## 📅 时间线

```
Christiano et al. (2017) → RLHF 原始方法
Stiennon et al. (2020) → RLHF 用于摘要
GPT-3 (2020) → 大但不对齐
    【InstructGPT (2022.03)】→ RLHF 通用化
ChatGPT (2022.11) → 对话优化版 → 爆发
Constitutional AI (2022.12) → AI 自我对齐
LLaMA-2 Chat (2023.07) → 开源 RLHF
DPO (2023.05) → 跳过 RM 的直接偏好优化
```

## ❓ 深度思考题

1. **概念题**：为什么 SFT 不够，还需要 RM+PPO？SFT 的局限在哪里？
2. **设计题**：如果让你设计 RLHF 的标注指南，你会强调哪些标准？怎么处理标注员之间的分歧？
3. **批判题**：InstructGPT 对齐了 40 人的偏好——这算"对齐"吗？怎么扩展到更广泛的人群？
4. **实践题**：为什么后来 DPO 能替代 RLHF？DPO 的优势是什么？局限是什么？
5. **扩展题**：如果 RLHF 的 RM 有系统性偏差（比如偏好冗长回答），怎么检测和修复？

## 📚 延伸阅读

| 论文 | 年份 | 关系 |
|------|------|------|
| Christiano et al. (RLHF) | 2017 | RLHF 原始方法 |
| ChatGPT | 2022 | InstructGPT 的对话版 |
| Constitutional AI | 2022 | AI 自我对齐 |
| DPO | 2023 | 跳过 RM 的偏好优化 |
| LLaMA-2 Chat | 2023 | 开源 RLHF |
