# 📚 LLM 论文精读教程

> 用 [llm-math-foundations](https://github.com/CCODING04/llm-math-foundations) 的数学基础，带你逐篇精读 LLM 领域的经典论文。

## 🎯 项目目标

1. **以数学为锚点**：每篇论文的学习都关联到 llm-math-foundations 中的数学知识
2. **Step by Step 教学**：不是论文翻译，而是真正的教学——有引导、有提问、有代码验证
3. **多维度阅读**：每篇论文提供独立讲解 + 论文原文融合讲解两种模式

## 📖 论文列表

### 第一阶段：基础架构

| # | 论文 | 年份 | 核心主题 | 状态 |
|---|------|------|---------|------|
| 01 | [Attention Is All You Need](./papers/01-attention-is-all-you-need/) | 2017 | Transformer 架构 | ✅ 完成 |

### 第二阶段：预训练范式

| # | 论文 | 年份 | 核心主题 | 面试重要度 | 状态 |
|---|------|------|---------|-----------|------|
| 02 | [BERT](./papers/02-bert/) | 2018 | 双向预训练编码器 | ⭐⭐⭐⭐⭐ | ✅ 完成 |
| 03 | [GPT-2](./papers/03-gpt2/) | 2019 | 自回归语言模型 | ⭐⭐⭐⭐ | 📋 待开始 |

### 第三阶段：规模化突破

| # | 论文 | 年份 | 核心主题 | 面试重要度 | 状态 |
|---|------|------|---------|-----------|------|
| 04 | [GPT-3](./papers/04-gpt3/) | 2020 | Few-shot / In-context Learning | ⭐⭐⭐⭐⭐ | 📋 待开始 |
| 05 | [Chinchilla](./papers/05-chinchilla/) | 2022 | Scaling Laws / 计算最优 | ⭐⭐⭐⭐ | 📋 待开始 |

### 第四阶段：对齐与对话

| # | 论文 | 年份 | 核心主题 | 面试重要度 | 状态 |
|---|------|------|---------|-----------|------|
| 06 | [InstructGPT](./papers/06-instructgpt/) | 2022 | RLHF 对齐 | ⭐⭐⭐⭐⭐ | 📋 待开始 |
| 07 | [LLaMA](./papers/07-llama/) | 2023 | 开源基座模型 | ⭐⭐⭐⭐ | 📋 待开始 |

### 第五阶段：效率优化

| # | 论文 | 年份 | 核心主题 | 面试重要度 | 状态 |
|---|------|------|---------|-----------|------|
| 08 | [LoRA](./papers/08-lora/) | 2021 | 参数高效微调 | ⭐⭐⭐⭐⭐ | 📋 待开始 |

### 🇨🇳 附加：国产前沿论文

| # | 论文 | 年份 | 核心主题 | 面试重要度 | 状态 |
|---|------|------|---------|-----------|------|
| A1 | [DeepSeek-V3](./papers/A1-deepseek-v3/) | 2024 | MoE + MLA 架构 | ⭐⭐⭐⭐⭐ | 📋 待开始 |
| A2 | [DeepSeek-R1](./papers/A2-deepseek-r1/) | 2025 | 纯 RL 推理能力 | ⭐⭐⭐⭐⭐ | 📋 待开始 |


---

## 📝 各论文阅读注意事项

### 02 - BERT
- **重点理解**：MLM（Masked Language Model）为什么能实现双向编码？与 GPT 的单向有什么本质区别？
- **面试常问**：BERT 的预训练任务（MLM + NSP）、[CLS] token 的作用、为什么 NSP 后续被 RoBERTa 证明没必要
- **关联**：和 GPT-2 对比阅读，理解编码器 vs 解码器路线的分歧

### 03 - GPT-2
- **重点理解**：自回归语言模型如何统一所有 NLP 任务（不需要监督微调）
- **面试常问**：GPT-2 的 zero-shot 方式、scaling 的核心论点（更大模型 = 更强能力）
- **关联**：衔接 GPT-3 的 few-shot learning

### 04 - GPT-3
- **重点理解**：In-context Learning 的本质——模型在不更新参数的情况下学习新任务
- **面试常问**：few-shot vs zero-shot vs fine-tuning 的区别、GPT-3 的规模和训练细节
- **注意**：论文很长（72页），重点读 Section 2-3（方法+评估），附录可略读

### 05 - Chinchilla
- **重点理解**：数据量 vs 模型大小的最优比例（约 20:1 的 token 数与参数量之比）
- **面试常问**：为什么 LLaMA 能用小模型打赢大模型？Chinchilla 最优意味着什么？
- **关联**：你 CS336 学的 scaling laws 可以直接关联

### 06 - InstructGPT
- **重点理解**：RLHF 三步流程（SFT → Reward Model → PPO）的每一步为什么需要
- **面试常问**：PPO 在 RLHF 中具体怎么用？Reward Model 怎么训练？什么是 reward hacking？
- **关联**：你的 llm-math-foundations Ch09 有 PPO 推导，可以互相印证

### 07 - LLaMA
- **重点理解**：Chinchilla 最优的实践验证——更多数据 + 更小模型的策略
- **面试常问**：训练数据配方（各来源占比）、RoPE、SwiGLU、RMSNorm 等结构改进
- **注意**：论文比较工程化，重点在训练细节和 scaling 实验

### 08 - LoRA
- **重点理解**：为什么低秩分解有效？ΔW = AB 的直觉
- **面试常问**：LoRA 的 rank 通常设多少？为什么可以很小？与全量微调的效果差距？
- **注意**：论文短且精，建议精读，公式推导要吃透

### A1 - DeepSeek-V3
- **重点理解**：MLA（Multi-head Latent Attention）如何压缩 KV Cache、MoE 的细粒度专家设计
- **面试常问**：为什么 MLA 比 MHA 省？负载均衡怎么做？
- **注意**：技术报告很长，重点读架构设计和训练方案

### A2 - DeepSeek-R1
- **重点理解**：纯 RL（不用 SFT）就能涌现 CoT 推理能力
- **面试常问**：R1-Zero 和 R1 的区别、蒸馏 vs RL 的效果对比
- **注意**：这是 2025 年最热的论文之一，面试加分利器

## 📂 每篇论文的文件结构

```
papers/XX-paper-name/
├── paper.pdf              # ① 论文原始 PDF
├── raw-extract.md         # ② MinerU 转换的 Markdown + 图片
├── images/                # ② 提取的论文图片
├── README.md              # ③ 独立中文讲解教程
├── merged-tutorial.md     # ④ 论文原文 + 讲解融合版（边看边学）
└── figures-analysis.md    # 图表分析（工作文件）
```

**四种阅读方式**：

| # | 文件 | 适合场景 |
|---|------|---------|
| ① | `paper.pdf` | 原始论文，引用参考 |
| ② | `raw-extract.md` | 论文 Markdown 版，方便搜索/引用 |
| ③ | `README.md` | 纯讲解版，系统学习时使用 |
| ④ | `merged-tutorial.md` | 边看论文边听讲解，沉浸式精读 |

## 🔗 关联资源

- **数学基础教程**：[llm-math-foundations](https://github.com/CCODING04/llm-math-foundations)
- **学习笔记**：[diy-llm-notes](https://github.com/CCODING04/diy-llm-notes)

## 🤖 教程生成流程

详见 [WORKFLOW.md](./WORKFLOW.md) — 添加新论文时按流程执行即可。
