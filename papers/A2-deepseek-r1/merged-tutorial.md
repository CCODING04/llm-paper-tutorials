# DeepSeek-R1 论文精读融合版

> 本文档以论文原文为骨架，在每个章节后插入讲解块。原文保持不动，讲解帮助理解。

---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

DeepSeek-AI

research@deepseek.com

# Abstract

General reasoning represents a long-standing and formidable challenge in artificial intelligence. Recent breakthroughs, exemplified by large language models (LLMs) (Brown et al., 2020; OpenAI, 2023) and chain-of-thought prompting (Wei et al., 2022b), have achieved considerable success on foundational reasoning tasks. However, this success is heavily contingent upon extensive human-annotated demonstrations, and models' capabilities are still insufficient for more complex problems. Here we show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories. The proposed RL framework facilitates the emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation. Consequently, the trained model achieves superior performance on verifiable tasks such as mathematics, coding competitions, and STEM fields, surpassing its counterparts trained via conventional supervised learning on human demonstrations. Moreover, the emergent reasoning patterns exhibited by these large-scale models can be systematically harnessed to guide and enhance the reasoning capabilities of smaller models.

> 📖 **讲解**
>
> 这段摘要用三句话概括了全文的核心：
>
> 1. **问题**：现有推理方法依赖大量人类标注，且对复杂问题能力不足
> 2. **方法**：用纯 RL 激发推理能力，不需要人类标注的推理轨迹
> 3. **发现**：模型自主涌现了 self-reflection、verification 等高级推理行为
>
> 最关键的一个词是 **"incentivized"**（激励）——不是"教会"模型推理，而是"激励"它自己发展出推理能力。这个用词反映了 R1 的核心哲学：**不教怎么做，只告诉对不对**。
>
> 还有一个重要细节：摘要最后提到"reasoning patterns can be systematically harnessed to guide smaller models"——这预告了蒸馏策略，用大模型涌现的推理模式来教小模型。

---

# 1. Introduction

Reasoning capability, the cornerstone of human intelligence, enables complex cognitive tasks ranging from mathematical problem-solving to logical deduction and programming. Recent advances in artificial intelligence have demonstrated that large language models (LLMs) can exhibit emergent behaviors, including reasoning abilities, when scaled to a sufficient size (Kaplan et al., 2020; Wei et al., 2022a). However, achieving such capabilities in pre-training typically demands substantial computational resources. In parallel, a complementary line of research has demonstrated that large language models can be effectively augmented through chain-ofthought (CoT) prompting. This technique, which involves either providing carefully designed few-shot examples or using minimalistic prompts such as "Let's think step by step"(Kojima et al., 2022; Wei et al., 2022b), enables models to produce intermediate reasoning steps, thereby substantially enhancing their performance on complex tasks. Similarly, further performance gains have been observed when models learn high-quality, multi-step reasoning trajectories during the post-training phase (Chung et al., 2024; OpenAI, 2023). Despite their effectiveness, these approaches exhibit notable limitations. Their dependence on human-annotated reasoning traces hinders scalability and introduces cognitive biases. Furthermore, by constraining models to replicate human thought processes, their performance is inherently capped by the humanprovided exemplars, which prevents the exploration of superior, non-human-like reasoning pathways.

> 📖 **讲解**
>
> 引言第一段建立了完整的背景脉络：
>
> | 路线 | 代表工作 | 核心思想 | 局限 |
> |------|---------|---------|------|
> | 规模涌现 | GPT-3, Chinchilla | 模型够大就能推理 | 计算成本极高 |
> | Prompt 工程 | CoT, "Let's think step by step" | 引导模型展示中间步骤 | 依赖人类设计的 prompt |
> | 后训练 SFT | InstructGPT, FLAN | 学习人类推理轨迹 | **上限被人类示范限制** |
>
> 最重要的一句话是最后那句——"constraining models to replicate human thought processes... prevents the exploration of superior, non-human-like reasoning pathways"。这是 R1 存在的根本理由：**人类的推理方式不一定是最好的推理方式**。

To tackle these issues, we aim to explore the potential of LLMs for developing reasoning abilities through self-evolution in an RL framework, with minimal reliance on human labeling efforts. Specifically, we build upon DeepSeek-V3-Base (DeepSeek-AI, 2024b) and employ Group Relative Policy Optimization (GRPO) (Shao et al., 2024) as our RL framework. The reward signal is solely based on the correctness of final predictions against ground-truth answers, without imposing constraints on the reasoning process itself. Notably, we bypass the conventional supervised fine-tuning (SFT) phase before RL training. This design choice stems from our hypothesis that human-defined reasoning patterns may limit model exploration, whereas unrestricted RL training can better incentivize the emergence of novel reasoning capabilities in LLMs. Through this process, detailed in Section 2, our model (referred to as DeepSeek-R1- Zero) naturally developed diverse and sophisticated reasoning behaviors. In solving reasoning problems, the model exhibits a tendency to generate longer responses, incorporating verification, reflection, and the exploration of alternative approaches within each response. Although we do not explicitly teach the model how to reason, it successfully learns improved reasoning strategies through reinforcement learning.

> 📖 **讲解**
>
> 这段揭示了 R1-Zero 的核心设计：
>
> - **基座**：DeepSeek-V3-Base（不是 V3 instruction-tuned！）
> - **算法**：GRPO（不需要 Value Model）
> - **奖励**：只看最终答案对不对（outcome-based）
> - **关键决策**：跳过 SFT，直接 RL
>
> "we bypass the conventional supervised fine-tuning (SFT) phase"——这句话是 R1-Zero 和之前所有 RLHF 工作的根本区别。InstructGPT 是 SFT → RLHF，R1-Zero 是直接 RL。
>
> 还有一个重要观察：模型自发学会了生成更长的回答（longer responses），包含了验证（verification）、反思（reflection）、和探索替代方案。这些行为**没有被显式训练**，是涌现的。

Although DeepSeek-R1-Zero demonstrates excellent reasoning capabilities, it faces challenges such as poor readability and language mixing, occasionally combining English and Chinese within a single chain-of-thought response. Furthermore, the rule-based RL training stage of DeepSeek-R1-Zero is narrowly focused on reasoning tasks, resulting in limited performance in broader areas such as writing and open-domain question answering. To address these challenges, we introduce DeepSeek-R1, a model trained through a multi-stage learning framework that integrates rejection sampling, reinforcement learning, and supervised finetuning, detailed in Section 3. This training pipeline enables DeepSeek-R1 to inherit the reasoning capabilities of its predecessor, DeepSeek-R1-Zero, while aligning model behavior with human preferences through additional non-reasoning data.

> 📖 **讲解**
>
> R1-Zero 的三个问题：
> 1. **可读性差**：推理过程像原始思维流，不像人类精心组织的答案
> 2. **语言混合**：中文问题用英文思考，或中英夹杂
> 3. **写作能力有限**：rule-based reward 只能验证数学/代码，无法评估写作质量
>
> R1 完整版就是解决这些问题的：多阶段训练 + rejection sampling + SFT + 非推理数据。
> 这里的 trade-off 很明确：**推理纯度 vs 用户体验**。

To enable broader access to powerful AI at lower energy cost, we have distilled several smaller models and made them publicly available. These distilled models exhibit strong reasoning capabilities, surpassing the performance of their original instruction-tuned counterparts. We believe that these instruction-tuned versions will also significantly contribute to the research community by providing a valuable resource for understanding the mechanisms underlying long chain-of-thought (CoT) reasoning models and for fostering the development of more powerful reasoning models. We release DeepSeek-R1 series models to the public at https://huggingface.co/deepseek-ai.

> 📖 **讲解**
>
> 蒸馏是 R1 项目的另一个重大贡献。6 个蒸馏模型（1.5B~70B）全部开源，让任何人都能在本地部署推理模型。
>
> 特别注意：1.5B 蒸馏模型在 MATH-500 上达到 83.9%，超过 GPT-4o 的 74.6%。这意味着**推理能力可以通过蒸馏有效迁移到极小的模型**。

---

# 2. DeepSeek-R1-Zero

We begin by elaborating on the training of DeepSeek-R1-Zero, which relies exclusively on reinforcement learning without supervised fine-tuning. To facilitate large-scale RL efficiency, we adopt Group Relative Policy Optimization (GRPO) (Shao et al., 2024).

> 📖 **讲解**
>
> R1-Zero 是论文最核心的实验。它回答了一个根本问题：**一个未经 SFT 的 base model，能仅通过 RL 学会推理吗？** 答案是肯定的。

## 2.1. Group Relative Policy Optimization

GRPO (Shao et al., 2024) is the reinforcement learning algorithm that we adopt to train DeepSeek-R1-Zero and DeepSeek-R1. It was originally proposed to simplify the training process and reduce the resource consumption of Proximal Policy Optimization (PPO) (Schulman et al., 2017), which is widely used in the RL stage of LLMs (Ouyang et al., 2022).

For each question $q ,$ GRPO samples a group of outputs $\{ o 1 , o 2 , \cdots , o _ { G } \}$ from the old policy $\pi _ { \theta _ { o l d } }$ and then optimizes the policy model $\pi _ { \theta }$ by maximizing the following objective:

$$
\mathcal {J} _ {G R P O} (\theta) = \mathbb {E} [ q \sim P (Q), \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {o l d}} (O | q) ]
$$

$$
\frac {1}{G} \sum_ {i = 1} ^ {G} \left(\min \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta _ {o l d}} (o _ {i} | q)} A _ {i}, \operatorname{clip} \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta _ {o l d}} (o _ {i} | q)}, 1 - \varepsilon , 1 + \varepsilon\right) A _ {i}\right) - \beta \mathbb {D} _ {K L} \left(\pi _ {\theta} | | \pi _ {r e f}\right)\right), \tag {1}
$$

$$
\mathbb {D} _ {K L} \left(\pi _ {\theta} | | \pi _ {r e f}\right) = \frac {\pi _ {r e f} (o _ {i} | q)}{\pi _ {\theta} (o _ {i} | q)} - \log \frac {\pi _ {r e f} (o _ {i} | q)}{\pi _ {\theta} (o _ {i} | q)} - 1, \tag {2}
$$

where $\pi _ { r e f }$ is a reference policy, ?? and ?? are hyper-parameters, and $A _ { i }$ is the advantage, computed using a group of rewards $\{ r _ { 1 } , r _ { 2 } , \ldots , r _ { G } \}$ corresponding to the outputs within each group:

$$
A _ {i} = \frac {r _ {i} - \text { mean } (\{r _ {1} , r _ {2} , \cdots , r _ {G} \})}{\text { std } (\{r _ {1} , r _ {2} , \cdots , r _ {G} \})}. \tag {3}
$$

> 📖 **讲解**
>
> 这是 GRPO 的完整公式。逐行拆解：
>
> **Eq. 1（目标函数）**：
> - 对每个问题 q，采样 G=16 个输出
> - 对每个输出，计算 clipped surrogate objective + KL 惩罚
> - 取 G 个输出的平均
>
> **Clipped Surrogate**：
> - ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)：新旧策略的概率比
> - clip(ρ_i, 1-ε, 1+ε)：限制策略变化幅度
> - min(ρ_i * A_i, clip(ρ_i) * A_i)：取较小值，保守更新
>
> **Eq. 2（KL 散度估计）**：
> - 这是 Schulman 2020 提出的无偏估计
> - 关键：KL 项直接加在 loss 里，不是作为 per-token reward
> - 这意味着 KL 惩罚与响应长度无关——不会阻止模型生成更长的推理
>
> **Eq. 3（Advantage）**：
> - 最核心的创新！
> - A_i = (r_i - mean) / std：就是 Z-score 标准化
> - 答对的（r=1）比平均好 → 正 advantage → 增大概率
> - 答错的（r=0）比平均差 → 负 advantage → 减小概率
> - 不需要 Value Model 来估计 baseline！

We give a comparison of GRPO and PPO in Supplementary A.3. To train DeepSeek-R1-Zero, we set the learning rate to 3e-6, the KL coefficient to 0.001, and the sampling temperature to 1 for rollout. For each question, we sample 16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward. As a result, both the performance and response length of DeepSeek-R1-Zero exhibit a significant jump at the 8.2k step, with training continuing for a total of 10,400 steps, corresponding to 1.6 training epochs. Each training step consists of 32 unique questions, resulting in a training batch size of 512. Every 400 steps, we replace the reference model with the latest policy model. To accelerate training, each rollout generates 8,192 outputs, which are randomly split into 16 mini-batches and trained for only a single inner epoch.

> 📖 **讲解**
>
> 关键训练细节：
>
> | 参数 | 值 | 意义 |
> |------|-----|------|
> | Learning Rate | 3e-6 | 标准 RL 学习率，比预训练小很多 |
> | KL Coefficient (β) | 0.001 | 很小的 KL 约束，允许策略大幅探索 |
> | Temperature | 1.0 | 高温，鼓励多样性 |
> | G（每题采样数） | 16 | 组内 16 个样本计算 advantage |
> | Max Length | 32K → 64K | 8.2K 步后翻倍，给模型更多思考空间 |
> | 训练步数 | 10,400（1.6 epochs） | 相当少的训练轮数 |
> | 参考模型更新 | 每 400 步 | 保持 KL 约束的相关性 |
>
> **8.2K 步的跳跃**：这是本文一个有趣的发现——当 max length 从 32K 扩展到 64K 时，AIME 性能和响应长度都出现了显著跳跃。这说明推理能力受到"思考空间"的限制——给模型更多的 token 预算，它就能思考得更深入。

Table 1 | Template for DeepSeek-R1-Zero.

> 📖 **讲解**
>
> 训练模板极其简单——只要求模型先在 `<think` 标签内思考，再在 `<answer` 标签内给答案。没有任何内容约束。
>
> 这意味着模型完全自由决定"怎么想"。这个极简设计是为了观察 RL 训练中推理行为的自然涌现。

## 2.2. Reward Design

The reward is the source of the training signal, which decides the direction of RL optimization. For DeepSeek-R1-Zero, we employ rule-based rewards to deliver precise feedback for data in mathematical, coding, and logical reasoning domains. Our rule-based reward system mainly consists of two types of rewards: accuracy rewards and format rewards.

> 📖 **讲解**
>
> 奖励设计哲学：**只用 rule-based reward，不用 neural reward model**。
>
> 两种奖励：
> 1. **Accuracy Reward**：数学用 SymPy 解析比较，代码用编译器+测试用例
> 2. **Format Reward**：是否正确使用 `<think` `<answer` 标签
>
> 为什么不用 neural RM？
> 1. 大规模 RL 中 neural RM 容易 reward hacking
> 2. 重新训练 RM 需要大量计算
> 3. Rule-based reward 对数学/代码来说足够精确

Format rewards complement the accuracy reward model by enforcing specific formatting requirements. In particular, the model is incentivized to encapsulate its reasoning process within designated tags, specifically '<think\>' and '</think\>'. This ensures that the model's thought process is explicitly delineated, enhancing interpretability and facilitating subsequent analysis.

$$
R e w a r d _ {\text { rule }} = R e w a r d _ {\text { acc }} + R e w a r d _ {\text { format }} \tag {4}
$$

The accuracy, reward and format reward are combined with the same weight. Notably, we abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks. This decision is predicated on our observation that neural reward models are susceptible to reward hacking during large-scale reinforcement learning. Moreover, retraining such models necessitates substantial computational resources and introduces additional complexity into the training pipeline, thereby complicating the overall optimization process.

## 2.3. Incentivize Reasoning Capability in LLMs

Specifically, we apply the RL technique on the DeepSeek-V3 base to train DeepSeek-R1-Zero. During training, we design a straightforward template, to require DeepSeek-R1-Zero to first produce a reasoning process, followed by the final answer. We intentionally limit our constraints to this structural format, avoiding any content-specific biases to ensure that we can accurately observe the model's natural progression during the RL process.

Figure 1(a) depicts the performance trajectory of DeepSeek-R1-Zero on the AIME 2024 benchmark throughout the RL training process, where the average pass@1 score on AIME 2024 shows a significant increase, jumping from an initial 15.6% to 77.9%. In addition, by leveraging the self-consistency decoding (Wang et al., 2023c), the model's performance can be further improved, achieving an accuracy of 86.7%. This performance significantly surpasses the average performance across all human competitors. Besides the math competitions, as shown in Figure 10, DeepSeek-R1-Zero also achieves remarkable performance in coding competitions and graduate-level biology, physics, and chemistry problems. These results underscore the effectiveness of RL in enhancing the reasoning capabilities of large language models.

> 📖 **讲解**
>
> R1-Zero 的核心结果：
> - AIME Pass@1: 15.6% → 77.9%（纯 RL，无 SFT）
> - Cons@16: 86.7%（16 个答案取多数投票）
> - 人类平均：37.8%
>
> 这个结果的意义在于：**没有人教模型怎么推理，它自己学会了**。而且推理水平远超人类竞赛选手平均。
>
> 注意 Cons@16 的提升（77.9% → 86.7%）：即使单次推理已经很强，多次采样+投票还能进一步提升，说明模型在不确定时有不同的推理路径。

Table 2 | An interesting "aha moment" of an intermediate version of DeepSeek-R1-Zero.

> 📖 **讲解**
>
> 这是论文中最著名的 "Aha Moment"。模型在求解一个方程时，突然说：
>
> "Wait, wait. Wait. That's an aha moment I can flag here. Let's reevaluate this step-by-step..."
>
> 没有人教它用这种语气，没有人教它在出错时停下来反思。这是 RL 训练中**自发涌现**的行为。
>
> 论文原文说 "This is also an aha moment for us, allowing us to witness the power and beauty of reinforcement learning"——作者们也被这个现象震惊了。

The self-evolution of DeepSeek-R1-Zero exemplifies how RL can autonomously enhance a model's reasoning capabilities.

As shown in Figure 1(b), DeepSeek-R1-Zero exhibits a steady increase in thinking time throughout training, driven solely by intrinsic adaptation rather than external modifications. Leveraging long CoT, the model progressively refines its reasoning, generating hundreds to thousands of tokens to explore and improve its problem-solving strategies.

> 📖 **讲解**
>
> Figure 1(b) 展示了另一个关键发现：**响应长度从 ~1000 tokens 增长到 ~17500 tokens**。
>
> 这不是人为设定的——没有人告诉模型"你要写更长的回答"。模型自己发现更长的推理链有助于得到正确答案，因此 RL 优化自然引导模型增加推理长度。
>
> 这是 test-time compute scaling 的涌现形式：模型学会了"遇到难题就多想想"。

The increase in thinking time fosters the autonomous development of sophisticated behaviors. Specifically, DeepSeek-R1-Zero increasingly exhibits advanced reasoning strategies such as reflective reasoning and systematic exploration of alternative solutions (see Figure 9(a) in Supplementary C.2 for details), significantly boosting its performance on verifiable tasks like math and coding. Notably, during training, DeepSeek-R1-Zero exhibits an "aha moment" (Table 2), characterized by a sudden increase in the use of the word "wait" during reflections (see Figure 9(b) in Supplementary C.2 for details). This moment marks a distinct change in reasoning patterns and clearly shows the self-evolution process of DeepSeek-R1-Zero.

> 📖 **讲解**
>
> 涌现行为的时间线：
>
> | 阶段 | 行为变化 |
> |------|---------|
> | 0-3000 步 | 基础推理，短回答 |
> | 3000-5000 步 | 开始反思，但"wait"频率低 |
> | 5000-8000 步 | 反思词汇频率上升，自我验证出现 |
> | 8000-10400 步 | "wait" 激增，反思词汇频率达到 5-7x |
>
> 反思词汇包括："wait"、"mistake"、"however"、"but"、"retry"、"error"、"verify"、"wrong"、"evaluate"、"check"。

Figure 2 | The multi-stage pipeline of DeepSeek-R1.

> 📖 **讲解**
>
> Figure 2 展示了 R1 的完整训练流水线：
>
> 1. **R1-Zero 路径**（上方）：V3-Base → RL → R1-Zero → Sampling → 过滤 → 数据
> 2. **R1 路径**（下方）：Cold Start SFT → RL Stage 1 → Rejection Sampling + SFT → RL Stage 2 → R1
>
> 这张图的关键信息：
> - R1-Zero 是一条独立路径，证明了纯 RL 的可行性
> - R1 是在 R1-Zero 基础上的工程优化
> - 蒸馏数据来自 R1 的 rejection sampling 阶段

---

# 3. DeepSeek-R1

Although DeepSeek-R1-Zero exhibits strong reasoning capabilities, it faces several issues. DeepSeek-R1-Zero struggles with challenges like poor readability, and language mixing, as DeepSeek-V3-Base is trained on multiple languages, especially English and Chinese. To address these issues, we develop DeepSeek-R1, whose pipeline is illustrated in Figure 2.

In the initial stage, we collect thousands of cold-start data that exhibits a conversational, human-aligned thinking process. RL training is then applied to improve the model performance with the conversational thinking process and language consistency. Subsequently, we apply rejection sampling and SFT once more. This stage incorporates both reasoning and nonreasoning datasets into the SFT process, enabling the model to not only excel in reasoning tasks but also demonstrate advanced writing capabilities. To further align the model with human preferences, we implement a secondary RL stage designed to enhance the model's helpfulness and harmlessness while simultaneously refining its reasoning capabilities.

> 📖 **讲解**
>
> R1 的四阶段流水线详解：
>
> **阶段 0：Cold Start SFT**
> - 数据来源：从 R1-Zero 生成正确答案，人工改写为可读的长 CoT，再用 V3 扩展
> - 数据量：数千条（非常少！）
> - 目的：给 RL 一个好的起点（比 raw base model 更"像人"的推理风格）
>
> **阶段 1：Reasoning-focused RL**
> - 与 R1-Zero 类似，但加了 language consistency reward
> - 解决语言混合问题
> - Trade-off：语言一致性会略微降低推理性能（消融实验证实）
>
> **阶段 2：Rejection Sampling + SFT（800K 数据）**
> - 从阶段 1 checkpoint 做多次采样，只保留正确答案
> - 600K 推理数据 + 200K 非推理数据
> - 非推理数据包括写作、事实问答、翻译等
>
> **阶段 3：All-scenario RL**
> - 混合推理 + 通用数据
> - 推理数据用 rule-based reward，通用数据用 neural RM
> - Temperature 降到 0.7
> - Neural RM 只在最后 400 步引入（防止 reward hacking）

## 3.1. Model-based Rewards

For general data, we resort to reward models to capture human preferences in complex and nuanced scenarios. We build upon the DeepSeek-V3 pipeline and adopt a similar distribution of preference pairs and training prompts. For helpfulness, we focus exclusively on the final summary, ensuring that the assessment emphasizes the utility and relevance of the response to the user while minimizing interference with the underlying reasoning process. For harmlessness, we evaluate the entire response of the model, including both the reasoning process and the summary, to identify and mitigate any potential risks, biases, or harmful content that may arise during the generation process.

> 📖 **讲解**
>
> Neural Reward Model 的设计：
>
> | 维度 | 评估范围 | 训练数据 |
> |------|---------|---------|
> | Helpfulness | 只看最终 summary | 66K preference pairs |
> | Harmlessness | 看完整响应（推理+summary） | 106K safe/unsafe 标注 |
>
> 关键设计决策：**helpfulness 只评估 summary，不评估推理过程**。这是因为推理过程本身可能很冗长混乱，但不影响最终答案的有用性。

## 3.2. Training Details

### 3.2.1. Training Details of the First RL Stage

In the first stage of RL, we set the learning rate to 3e-6, the KL coefficient to 0.001, the GRPO clip ratio ?? to 10, and the sampling temperature to 1 for rollout.

$$
\text { Reward } _ {\text { language }} = \frac {\text { Num } (\text { Words } _ {\text { target }})}{\text { Num } (\text { Words })} \tag {7}
$$

Although ablation experiments in Supplementary B.6 show that such alignment results in a slight degradation in the model's performance, this reward aligns with human preferences, making it more readable.

> 📖 **讲解**
>
> Language Consistency Reward 的 trade-off：
> - **好处**：推理过程保持语言一致，用户更易读
> - **代价**：推理性能略微下降（Figure 7 的消融实验证实）
> - 消融实验显示：数学 benchmark 差异不大，但代码 benchmark 有明显差距
> - 这是一个**对齐税**（alignment tax）的例子：让模型更"人友好"，需要牺牲一些能力

### 3.2.2. Training Details of the Second RL Stage

Specifically, we train the model using a combination of reward signals and diverse prompt distributions.

$$
R e w a r d = R e w a r d _ {\text {reasoning}} + R e w a r d _ {\text {general}} + R e w a r d _ {\text {language}} \tag {8}
$$

The second stage of RL retains most of the parameters from the first stage, with the key difference being a reduced temperature of 0.7, as we find that higher temperatures in this stage lead to incoherent generation. The stage comprises a total of 1,700 training steps, during which general instruction data and preference-based rewards are incorporated exclusively in the final 400 steps. We find that more training steps with the model based preference reward signal may lead to reward hacking, which is documented in Supplementary B.5.

> 📖 **讲解**
>
> 第二阶段 RL 的关键设计：
>
> - Temperature 从 1.0 降到 0.7：防止高温导致推理不连贯
> - Neural RM 只在最后 400 步引入：更多步数会导致 reward hacking
> - Figure 6 展示了 reward hacking 的现象：reward 持续上升但实际性能下降
>
> **Reward Hacking 的教训**：模型会找到"欺骗" reward model 的方式来获取高分，而不是真正提升质量。这是 neural RM 的根本挑战，也是 R1 在推理任务上坚持用 rule-based reward 的原因。

---

# 4. Experiment

We evaluate our models on MMLU, MMLU-Redux, MMLU-Pro, C-Eval, CMMLU, IFEval, FRAMES, GPQA Diamond, SimpleQA, C-SimpleQA, SWE-Bench Verified, Aider, LiveCodeBench, Codeforces, CNMO 2024, and AIME 2024.

> 📖 **讲解**
>
> 评估覆盖了 5 大维度：
> 1. **通用知识**：MMLU/MMLU-Pro/C-Eval
> 2. **推理**：AIME/MATH-500/CNMO/GPQA
> 3. **编程**：LiveCodeBench/Codeforces/SWE-Bench/Aider
> 4. **指令遵循**：IFEval/FRAMES
> 5. **用户偏好**：AlpacaEval 2.0/ArenaHard

Table 3 | Experimental results at each stage of DeepSeek-R1.

> 📖 **讲解**
>
> 各阶段的进化：
>
> | 指标 | R1-Zero → Dev1 → Dev2 → Dev3 → R1 |
> |------|----------------------------------|
> | AIME | 77.9 → 59.0↓ → 74.0↑ → 78.1↑ → 79.8↑ |
> | ArenaHard | 53.6 → 77.0↑ → 73.2 → 75.6 → **92.3**↑ |
> | MATH-500 | 95.9 → 94.2↓ → 95.9 → 95.4 → **97.3** |
>
> **关键观察**：
> - Cold Start (Dev1) 让 AIME 下降（77.9→59.0），因为少量 SFT 数据限制了推理多样性
> - RL Stage 1 (Dev2) 恢复了推理能力
> - SFT (Dev3) 提升了通用能力（AlpacaEval↑）
> - 最终 RL (R1) 让 ArenaHard 从 75.6 飙升到 92.3
>
> 整个流水线是一个"推理→可读性→推理恢复→通用提升→最终对齐"的迭代过程。

Table 8 | Comparison between DeepSeek-R1 and other representative models.

> 📖 **讲解**
>
> 核心对比结果：
>
> | Benchmark | R1 | o1-1217 | o1-mini | GPT-4o | V3 |
> |-----------|:-:|:-:|:-:|:-:|:-:|
> | AIME 2024 | 79.8 | 79.2 | 63.6 | 9.3 | 39.2 |
> | MATH-500 | **97.3** | 96.4 | 90.0 | 74.6 | 90.2 |
> | Codeforces Rating | 2029 | **2061** | 1820 | 759 | 1134 |
> | MMLU-Pro | **84.0** | - | 80.3 | 72.6 | 75.9 |
> | GPQA Diamond | 71.5 | **75.7** | 60.0 | 49.9 | 59.1 |
>
> R1 的定位很清晰：
> - 数学/编程：与 o1-1217 持平或接近
> - 通用知识：MMLU-Pro 显著领先
> - 博士级推理（GPQA）：不如 o1，可能因为缺乏工具使用
> - 成本：$294K（不到 o1 估计成本的 1/30）

Figure 10 | The benchmark performance of DeepSeek-R1 and DeepSeek-R1-Zero is compared with human scores.

> 📖 **讲解**
>
> Figure 10 展示了 R1 vs 人类的对比：
> - AIME: R1 (79.8) >> 人类平均 (37.8)
> - Codeforces: R1 超过 96.3% 的人类参赛者
> - GPQA: 人类博士+网络 (81.2) > R1 (71.5)
>
> GPQA 的差距很有意义——如果允许 R1 使用网络搜索，可能能缩小这个差距。论文也指出了这一点。

---

# 5. Ethics and Safety Statement

With the advancement in the reasoning capabilities of DeepSeek-R1, we deeply recognize the potential ethical risks.

> 📖 **讲解**
>
> 安全部分的关键发现：
> - R1 的内在安全水平中等（与 GPT-4o 相当）
> - 加上 risk control system 后安全水平大幅提升
> - R1 对 jailbreak 攻击较脆弱（unsafe rate 从 25.2% 飙升到 85.9%），需要外部风险控制
> - 推理模型比非推理模型更依赖外部安全系统

---

# 6. Conclusion, Limitation, and Future Work

We present DeepSeek-R1-Zero and DeepSeek-R1, which rely on large-scale RL to incentivize model reasoning behaviors. Our results demonstrate that pre-trained checkpoints inherently possess substantial potential for complex reasoning tasks. We believe that the key to unlocking this potential lies not in large-scale human annotation but in the provision of hard reasoning questions, a reliable verifier, and sufficient computational resources for reinforcement learning.

> 📖 **讲解**
>
> 结论的三个关键要素：
> 1. **难题**（hard reasoning questions）
> 2. **可靠的验证器**（reliable verifier = rule-based reward）
> 3. **足够的计算资源**（sufficient compute for RL）
>
> 这三个要素替代了传统方法对大规模人类标注的依赖。

Even if DeepSeek-R1 achieves frontier results on reasoning benchmarks, it still faces several capability limitations, as outlined below:

Structure Output and Tool Use: Currently, the structural output capabilities of DeepSeek-R1 remain suboptimal compared to existing models. Moreover, DeepSeek-R1 cannot leverage tools, such as search engines and calculators, to improve the performance of output.

Token efficiency: Unlike conventional test-time computation scaling approaches, such as majority voting or Monte Carlo Tree Search (MCTS), DeepSeek-R1 dynamically allocates computational resources during inference according to the complexity of the problem at hand. Nevertheless, there remains room for further optimization in terms of token efficiency, as instances of excessive reasoning—manifested as overthinking—are still observed in response to simpler questions.

Language Mixing: DeepSeek-R1 is currently optimized for Chinese and English, which may result in language mixing issues when handling queries in other languages.

Prompting Engineering: When evaluating DeepSeek-R1, we observe that it is sensitive to prompts. Few-shot prompting consistently degrades its performance. Therefore, we recommend users directly describe the problem and specify the output format using a zero-shot setting for optimal results.

Software Engineering Tasks: Due to the long evaluation times, which impact the efficiency of the RL process, large-scale RL has not been applied extensively in software engineering tasks.

Reward Hacking: The success of pure RL depends on reliable reward signals. If the reward signal is assigned by a model instead of predefined rules, it becomes more susceptible to exploitation as training progresses.

> 📖 **讲解**
>
> R1 坦诚地列出了所有局限。面试中如果被问到"R1 的不足"，这些就是标准答案：
>
> 1. **工具使用缺失**：不能调用搜索引擎/计算器
> 2. **Token 效率**：简单问题也会过度推理（overthinking）
> 3. **语言混合**：非中英文问题可能用英文推理
> 4. **Prompt 敏感**：few-shot 反而降低性能
> 5. **软件工程**：RL 没有覆盖
> 6. **Reward Hacking**：Neural RM 不可靠
>
> 这些局限也为后续工作指明了方向：工具集成、token 预算控制、多语言优化等。

---

# F. DeepSeek-R1 Distillation

LLMs are energy-intensive, requiring substantial computational resources. To address this challenge, we adopt a model distillation approach. Specifically, we fine-tune open-source foundation models such as Qwen and LLaMA using a curated dataset comprising 800,000 samples generated with DeepSeek-R1. We find that models distilled from high-quality teacher outputs consistently outperform those trained directly on human-generated data.

Table 15 | Comparison of DeepSeek-R1 distilled models and other comparable models.

> 📖 **讲解**
>
> 蒸馏结果的亮点：
>
> | 模型 | 参数量 | AIME | MATH-500 |
> |------|--------|------|----------|
> | GPT-4o | ~1T? | 9.3 | 74.6 |
> | Claude 3.5S | ~? | 16.0 | 78.3 |
> | **R1-Distill-Qwen-1.5B** | **1.5B** | **28.9** | **83.9** |
> | R1-Distill-Qwen-7B | 7B | 55.5 | 92.8 |
> | R1-Distill-Qwen-32B | 32B | 72.6 | 94.3 |
>
> 1.5B 模型在 MATH 上超越 GPT-4o（83.9 vs 74.6）——这是本文最惊人的数字之一。
>
> 蒸馏只用了 SFT，没有 RL。这说明推理模式可以通过简单的 SFT 有效迁移。

## F.1. Distillation v.s. Reinforcement Learning

Table 16 shows that the 32B base model, after large-scale RL training, achieves performance on par with QwQ-32B-Preview. However, DeepSeek-R1-Distill-Qwen-32B, which is distilled from DeepSeek-R1, performs significantly better than Qwen2.5-32B-Zero across all benchmarks.

> 📖 **讲解**
>
> 这是蒸馏 vs 直接 RL 的关键对比：
>
> | 方法 | AIME Pass@1 |
> |------|-------------|
> | R1-Distill-Qwen-32B（蒸馏） | **72.6** |
> | Qwen2.5-32B-Zero（直接 RL） | 47.0 |
> | QwQ-32B-Preview | 50.0 |
>
> **结论**：在小模型上，蒸馏 >> 直接 RL。
>
> 原因：32B 模型容量不足以涌现复杂推理行为。推理能力的涌现需要更大的模型（至少 230B+），但可以通过蒸馏高效迁移到小模型。
>
> 这也是 R1 项目的实用价值所在——用大模型发现推理模式，用蒸馏让小模型也具备这些模式。

---

# G. Discussion

## G.1. Key Findings

The importance of base checkpoint: we experimented with smaller-scale models, specifically a 7B dense model and a 16B MoE model, as the foundational architectures for RL training. However, these configurations consistently failed to yield meaningful improvements.

> 📖 **讲解**
>
> 关键发现 #1：**Base Model 的容量至关重要**。
>
> - 7B Dense + RL → 失败
> - 16B MoE + RL → 失败
> - 32B Dense + RL → 有一定效果
> - 230B MoE + RL → 有效
> - 671B MoE + RL → 涌现推理行为
>
> 这说明推理能力的涌现有一个"最小模型规模门槛"。

The importance of verifiers: The effectiveness of DeepSeek-R1-Zero is highly contingent upon the reliability and fidelity of the reward signal used during training.

> 📖 **讲解**
>
> 关键发现 #2：**可靠的验证器是关键**。
>
> 两种可靠方法：
> 1. Rule-based RM：数学/代码的确定性验证
> 2. LLM-as-Judge：用 V3 判断答案正确性（适用于有简短答案的任务）
>
> 对于开放性任务（写作等），验证器不可靠 → RL 效果差 → 需要回退到 SFT。

Iterative pipeline: We propose a multi-stage training pipeline comprising both SFT and RL stages.

> 📖 **讲解**
>
> 关键发现 #3：**RL 和 SFT 缺一不可**。
>
> - RL → 发现推理模式，无法通过人类示范获得
> - SFT → 处理可靠奖励信号难以定义的任务（写作等）
> - 纯 RL → reward hacking + 非推理任务差
> - 纯 SFT → 无法探索超越人类的推理路径

## G.2. Unsuccessful Attempts

Process Reward Model (PRM): PRM has three main limitations: (1) challenging to define a fine-grain step; (2) determining intermediate step correctness is difficult; (3) reward hacking.

Monte Carlo Tree Search (MCTS): This approach encounters several challenges: (1) exponentially larger search space than chess; (2) value model is hard to train; (3) local optima.

> 📖 **讲解**
>
> 论文坦诚分享了两个失败尝试，这在顶会论文中非常罕见：
>
> **PRM 为什么失败**：
> - 推理过程的"步骤"边界模糊
> - 中间步骤的对错难以自动判断
> - 引入 PRM 导致 reward hacking
>
> **MCTS 为什么失败**：
> - 语言生成的搜索空间远大于围棋
> - Value Model 难以准确估计中间状态价值
> - 节点扩展限制导致局部最优
>
> 这些失败经验对社区非常有价值——帮助后来者避免重复踩坑。

---

# Appendix A.3. A Comparison of GRPO and PPO

Figure 3 | Demonstration of PPO and our GRPO. GRPO foregoes the value model, instead estimating the advantages from group scores.

> 📖 **讲解**
>
> Figure 3 直观对比了 PPO 和 GRPO：
>
> **PPO**：4 个模型
> - Policy Model（正在训练的策略）
> - Value Model（估计每个状态的 baseline）
> - Reference Model（KL 约束参考）
> - Reward Model（计算奖励）
>
> **GRPO**：3 个模型（省掉 Value Model）
> - Policy Model
> - Reference Model
> - Reward Model（只需 rule-based）
>
> 省掉 Value Model 的代价：每个问题需要采样 G=16 次（更多计算），但换来显存减半。
>
> 在 671B 模型上，少一个同大小的模型意味着节省 ~336B 参数的显存。

Figure 4 | Performance of PPO and GRPO on the MATH task.

> 📖 **讲解**
>
> Figure 4 用 DeepSeek-Coder-V2-Lite (16B MoE) 在 MATH 上对比：
>
> - PPO (λ=0.95)：最终 ~50%
> - PPO (λ=1.0)：最终 ~53%
> - GRPO：最终 ~55%
>
> GRPO 略优于 PPO，但差距不大。GRPO 的真正优势不在于性能，而在于：
> 1. 不需要调 GAE 的 λ 超参数
> 2. 省掉 Value Model 的显存
> 3. 在大模型上更实际可行

---

# Appendix F. DeepSeek-R1 Distillation (continued)

Table 16 shows that distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation.

> 📖 **讲解**
>
> 最终总结：蒸馏 > 直接 RL（在小模型上）
>
> 但论文也指出：**要突破人类智能的边界，可能仍需要更强大的基础模型和更大规模的 RL**。
>
> 这意味着 R1 的蒸馏策略是一个权宜之计——当前最有效的方法，但长期来看，推理能力的根本提升还是需要大模型的 RL。
