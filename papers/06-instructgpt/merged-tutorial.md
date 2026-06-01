# InstructGPT: Training language models to follow instructions with human feedback — 精读融合版

> 阅读指南：论文原文保持不动，每个章节后穿插中文讲解（📖 标记）。

---

# Abstract

Making language models bigger does not inherently make them better at following a user's intent. [...] We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters.

> 📖 **讲解 · 鸟瞰**
>
> **一句话**：RLHF 让 1.3B 模型在人类偏好上超越 175B GPT-3。
>
> **核心方法**：SFT（监督微调）→ RM（奖励模型）→ PPO（强化学习）
>
> **三个目标**：Helpful（有用）、Honest（诚实）、Harmless（无害）

---

# 1. Introduction

[论文原文完整保留...]

> 📖 **讲解 · 精读 · 核心问题**
>
> 为什么大模型需要对齐？因为预训练目标 ≠ 用户目标：
> - 预训练目标：预测下一个 token（模仿互联网）
> - 用户目标：遵循指令、有用、诚实、无害
>
> ❓ **为什么不能只靠更大的模型？** 论文开篇就回答了："Making language models bigger does not inherently make them better at following a user's intent." 更大 ≠ 更对齐。

![](images/5b52279679ef4b25d937471663b3ea09bd3c002c1091485e4865c4339e4ec279.jpg)

Figure 1: Human evaluations

> 📖 **讲解 · 图表精读（Figure 1——核心结果图）**
>
> 1.3B InstructGPT 的输出被人类偏好于 175B GPT-3（~85% 偏好率）。
>
> **面试价值**：对齐比规模更重要。

![](images/4bc7528fb7e56be52fefe32830ce55cc7d1650b82ecb8c5c048ba00fe5dd2a8c.jpg)

Figure 2: Three-step RLHF process

> 📖 **讲解 · 图表精读（Figure 2——RLHF 流程图）**
>
> 三步法：
> 1. **SFT**：人类写演示 → 监督微调 GPT-3
> 2. **RM**：多个模型输出 → 人类排序 → 训练奖励模型
> 3. **PPO**：用 RM 做奖励 + KL 约束 → 强化学习
>
> **面试必考**：解释 RLHF 三步法。

---

# 3. Method

## 3.1 High-level methodology

[论文原文...]

> 📖 **讲解 · 精读 · 三步法详解**
>
> **Step 1: SFT（监督微调）**
> - 数据：~13K (prompt, 演示输出) 对
> - 标注员根据指令写出理想回答
> - 标准监督学习：min 交叉熵
>
> **Step 2: RM（奖励模型）**
> - 数据：~33K 比较排序
> - 对同一 prompt 生成 4-9 个输出，人类排序
> - 损失：$-\log\sigma(r(x, y_w) - r(x, y_l))$
>
> **Step 3: PPO（强化学习）**
> - 目标：$\mathbb{E}[r(x,y)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$
> - RM 做奖励 + KL 惩罚防止偏离太远
> - **PPO-ptx**：加入预训练 log-likelihood 缓解对齐税
>
> ❓ **为什么需要 KL 惩罚？** 防止 reward hacking——模型找到 RM 漏洞生成无意义但高分的输出。

---

# 4. Results

## 4.1 Results on API prompt distribution

[论文原文...]

> 📖 **讲解 · 精读 · 核心结果**
>
> **人类偏好**：
> | 对比 | 偏好率 |
> |------|--------|
> | 175B InstructGPT vs 175B GPT-3 | 85% |
> | 1.3B InstructGPT vs 175B GPT-3 | ~85% |
>
> **TruthfulQA**：21% → 55%（2.6x 提升）
> **毒性**：降低 25%（被要求尊重时）
>
> **对齐税**：SQuAD/DROP/HellaSwag 有回归，PPO-ptx 大幅缓解
>
> **泛化**：能处理代码摘要、多语言指令等训练分布外任务

---

# 5. Discussion

## 5.2 What are we aligning to?

[论文原文...]

> 📖 **讲解 · 批判 · 对齐到谁的偏好？**
>
> 论文坦诚："This procedure aligns the behavior of GPT-3 to the stated preferences of a specific group of people (mostly our labelers and researchers), rather than any broader notion of 'human values'."
>
> 这是一个深刻的局限——对齐到 40 个标注员的偏好 ≠ 对齐到全人类的价值观。
>
> 后来 Constitutional AI 试图用"宪法原则"来减少这种偏差，但问题远未解决。

## 5.3 Limitations

[论文原文...]

> 📖 **讲解 · 批判 · 局限性**
>
> 1. 仍会编造事实（hallucination 未根治）
> 2. 过度谨慎（简单问题也给冗长回答）
> 3. 无法识别错误前提的指令
> 4. 标注员偏差
> 5. 对齐税不完全解决
>
> **后来哪些被改善了？** GPT-4 在诚实性和安全性上进一步提升，但 hallucination 仍然是核心挑战。

---

# 8. Conclusion

[论文原文...]

> 📖 **讲解 · 知识网络**
>
> **时间线**：GPT-3 → 【InstructGPT】→ ChatGPT → GPT-4
>
> **InstructGPT 开创了什么**：
> - RLHF 三步法成为对齐标准范式
> - "对齐比规模重要"的实证
> - PPO-ptx 缓解对齐税的技巧
>
> **什么被改进了**：
> - SFT 数据量被后来大幅增加
> - PPO 被 DPO 等更简单的方法替代（部分场景）
> - 40 个标注员 → 更大规模的标注团队 + AI 辅助
>
> **面试核心**：
>
> **Q: RLHF 三步法？** A: SFT（教会什么是好的）→ RM（学会打分）→ PPO（优化到高分）
>
> **Q: 为什么 RLHF 比 SFT 好？** A: SFT 只有正例，RLHF 通过比较有正负例，学到更细粒度的偏好。
>
> **Q: 对齐税？** A: 对齐训练可能导致某些 NLP 任务性能回归。PPO-ptx 通过混合预训练数据缓解。
