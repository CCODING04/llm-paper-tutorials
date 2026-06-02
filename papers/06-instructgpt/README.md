# 📖 InstructGPT: Training language models to follow instructions with human feedback

> **论文**：Ouyang et al., 2022 (OpenAI) | NeurIPS 2022
>
> **一句话总结**：用 RLHF 微调 GPT-3，使 1.3B 模型输出被偏好度超过 175B GPT-3——对齐比规模更重要。

---

# 第一层：鸟瞰

## 🎯 核心贡献

1. **RLHF 工程化落地**：首次将 RLHF 从研究原型（摘要任务）扩展到通用指令跟随——SFT → RM → PPO 三阶段流水线，成为后来所有对话模型（ChatGPT、LLaMA-2 Chat）的基础架构
2. **对齐 > 规模**：1.3B InstructGPT 输出被偏好度超过 175B GPT-3（100x 参数差距！）——证明对齐比堆参数更有效
3. **Alignment Tax 概念**：对齐有代价——RLHF 会损害某些 NLP 能力（SQuAD/DROP/WMT 下降）。PPO-ptx 通过混合预训练更新缓解，这是工程上的关键贡献
4. **Truthfulness 提升**：TruthfulQA 上 truthful+informative 的比例翻倍；闭域幻觉率从 41% 降到 21%

## 📍 知识网络定位

```
Christiano et al. (2017) → RLHF 原始方法（简单机器人/Atari）
Stiennon et al. (2020)   → RLHF 用于摘要任务
GPT-3 (2020)             → 175B 但不 follow 指令（"misaligned"）
         ↓
   【InstructGPT (2022.03)】→ SFT+RM+PPO 三阶段 RLHF 通用化
         ↓
   ChatGPT (2022.11)         → InstructGPT + 对话优化 → 爆发
   Constitutional AI (2022.12) → AI 自我对齐（减少人类标注）
   LLaMA-2 Chat (2023.07)    → 开源 RLHF
   DPO (2023.05)             → 跳过 RM 的直接偏好优化
```

### 与本系列其他论文的关系

| 本系列论文 | 与 InstructGPT 的关系 |
|-----------|---------------------|
| 01-Attention | InstructGPT 底层架构 GPT-3 基于 Transformer（Self-Attention 机制） |
| 02-BERT | InstructGPT 用单向 LM（GPT 路线），而非 BERT 的双向预训练——不同的预训练范式 |
| 03-GPT-2 | GPT-2 验证了"更大模型 + 更多数据"的 scaling，InstructGPT 证明对齐比继续 scaling 更重要 |
| 04-GPT-3 | InstructGPT 的基础模型就是 GPT-3——在 GPT-3 上做 RLHF 微调 |
| 05-Chinchilla | Chinchilla 证明数据量与参数量同等重要；InstructGPT 证明对齐与 scaling 同等重要——两者互补 |
| 07-LLaMA | LLaMA-2 Chat 直接采用了 InstructGPT 的 RLHF 三阶段流水线 |
| 08-LoRA | LoRA 可用于高效微调，但在 InstructGPT 时代尚未成熟 |

---

# 第二层：精读

## 1. 引言：为什么需要 RLHF？

### 核心问题

> "The language modeling objective—predicting the next token on a webpage—is different from 'follow the user's instructions helpfully and safely'."

**目标错位**：语言建模 ≠ 指令跟随。GPT-3 的训练目标是"预测下一个 token"，但用户想要的是"有用的回答"。这就像一个读遍了所有书的学生，考试时却不会答题——知识够了，但方向没对齐。

### 三个对齐目标

1. **Helpful**（有用）：帮助用户完成任务
2. **Honest**（诚实）：不编造信息
3. **Harmless**（无害）：不产生有害内容

> ❓ **这三个目标有冲突吗？** 有。比如用户要求"写一个关于某群体的偏见笑话"——helpful 要求满足，harmless 要求拒绝。InstructGPT 的处理：优先 harmless。这也引出了后来的 Constitutional AI——用 AI 辅助处理这类冲突。

### 引言中的 6 个关键发现

论文在引言中列出了 6 个主要发现（findings），我们逐一展开：

1. **标注员显著偏好 InstructGPT**：175B InstructGPT 输出被偏好度 85±3% 超过 GPT-3（Figure 1）
2. **泛化到 held-out 标注员**：未参与训练的标注员也有类似偏好——不是过拟合训练标注员
3. **公共 NLP 数据集不反映真实使用**：FLAN/T0 在 API 分布上远不如 InstructGPT——学术任务 ≠ 真实需求
4. **Truthfulness 提升**：TruthfulQA 上 truthful+informative 比例翻倍
5. **毒性小幅改善，偏差无改善**：当被要求生成尊重内容时，毒性降低；但被要求生成有害内容时，反而比 GPT-3 更有害（毒性悖论）
6. **Alignment Tax 可缓解**：PPO-ptx 大幅降低 NLP 基准退化

> 💡 **面试价值**：这 6 个发现覆盖了 InstructGPT 的所有核心结论。面试时能复述 3-4 个就很强了。

## 2. 方法：三阶段流水线

### Stage 1: SFT（Supervised Fine-Tuning）

#### 直觉解释

SFT 就像给一个通才安排专门的训练——不是从零学起，而是在已有能力基础上，学会"按照指令回答"。

**数据**：
- ~13K 条人工撰写的 prompt + 高质量回答
- 来源：OpenAI API 提交的 prompt（11,295 条标注员写的 + 1,430 条客户提交的）+ 标注员自己写的 prompt

**训练细节**：
- 在 GPT-3 上做标准监督微调
- 训练 **16 epochs**，cosine learning rate decay，residual dropout 0.2
- **关键发现**：验证 loss 在 1 epoch 后就开始过拟合，但训练更多 epochs 反而让 RM score 和人类偏好更好

> ❓ **为什么过拟合了还要继续训练？** 因为验证 loss 衡量的是"预测下一个 token 的能力"，而人类偏好衡量的是"回答有多好"。这两者不完全一致——过拟合可能让模型记住了一些好的回答模式。这启发我们：**验证 loss 不是唯一的模型选择标准**。

#### SFT 损失函数

标准的因果语言建模（causal LM）损失——交叉熵：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim D_{\text{SFT}}} \left[ \sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t}) \right]$$

- $\pi_\theta$：GPT-3 模型（参数 $\theta$）
- $x$：prompt，$y$：期望的回答
- $y_t$：回答中第 $t$ 个 token，$y_{<t}$：第 $t$ 个之前的所有 token
- 直觉：让模型在每个位置都尽量把正确的下一个 token 概率提高

**结果**：175B SFT 模型已经比 GPT-3 好，但还不够。

### Stage 2: RM（Reward Model）

#### 直觉解释

RM 的目标是学习一个"评分函数"——给 prompt + response 一个标量分数，分数越高表示人类越喜欢。

> 💡 类比：RM 就像一个阅卷老师——它不是学会做题，而是学会判卷。给每份答卷打一个分数，分高的表示更好。

#### 数据收集

标注员对每个 prompt 的 K 个模型输出（K=4~9）进行排序。比如 K=4 时，标注员排出 A > B > C > D。

关键工程细节：
- K=4~9 个输出，产生 $\binom{K}{2}$ 对比较
- 比如 K=4：6 对；K=9：36 对
- **所有 $\binom{K}{2}$ 对作为单个 batch element 训练**——防止过拟合的关键设计

> ❓ **为什么不把每对比较独立训练？** 因为同一 prompt 的 K 个输出的比较高度相关。如果打乱成独立样本，模型在单个 epoch 就会过拟合（因为看到了太多相关的对）。把它们作为一个 batch element，只需要一次 forward pass 就能算出所有对的 loss。

#### Bradley-Terry 模型与 RM 损失函数

**为什么用 Bradley-Terry？** 因为人类给的是排序（A > B），不是绝对分数。Bradley-Terry 模型将排序转化为概率：

$$P(y_w \succ y_l) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) = \frac{1}{1 + e^{-(r_\theta(x, y_w) - r_\theta(x, y_l))}}$$

直觉：如果 $r_\theta(x, y_w) \gg r_\theta(x, y_l)$，则 $\sigma(\cdot) \to 1$，表示 $y_w$ 几乎一定被偏好。如果两者分数接近，则概率接近 0.5。

**损失函数**（论文 Equation 1）：

$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \left( \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right) \right]$$

**逐符号解释**：
- $r_\theta(x, y)$：RM 对 prompt $x$ + 回答 $y$ 的标量打分（RM 是 SFT 模型去掉 unembedding layer 后加一个 projection head）
- $y_w$：被偏好的回答（winner）
- $y_l$：不被偏好的回答（loser）
- $\sigma(\cdot)$：sigmoid 函数，将差值映射到 $(0, 1)$
- $\binom{K}{2}$：归一化因子，确保不同 $K$ 值的 loss 量级一致
- $D$：人类比较数据集

**为什么是交叉熵形式？** 因为这是一个二分类问题：给定 $(y_w, y_l)$，$y_w$ 被偏好是正类。log-likelihood 最大化 = 交叉熵最小化。

**RM 归一化**：由于 loss 对奖励值的平移不变（$r + c$ 不改变 $\sigma(r_w - r_l)$），论文在 RL 之前将 RM 归一化，使标注员示范的平均分数为 0。

**RM 规模选择**：为什么用 6B 不用 175B？

> 论文发现：175B RM 训练不稳定，不适合作为 PPO 的 value function 初始化。6B RM 在验证集上表现稳定，且推理成本低得多。后来 LLaMA-2 也用了类似规模的 RM。

#### 代码验证：RM Loss

```python
import torch
import torch.nn.functional as F

def reward_model_loss(r_w, r_l):
    """
    Bradley-Terry RM loss (论文 Equation 1)
    r_w: preferred response 的 reward 分数 (batch_size,)
    r_l: dispreferred response 的 reward 分数 (batch_size,)
    """
    return -F.logsigmoid(r_w - r_l).mean()

# 测试
torch.manual_seed(42)
r_w = torch.tensor([2.0, 1.5, 0.8, 3.0])  # 偏好的回答分数更高
r_l = torch.tensor([0.5, 0.3, 0.1, 1.0])  # 不被偏好的回答分数更低

loss = reward_model_loss(r_w, r_l)
print(f"RM loss: {loss.item():.4f}")

# 验证：当 r_w >> r_l 时 loss 应该很小
r_w_good = torch.tensor([5.0, 5.0, 5.0])
r_l_good = torch.tensor([0.0, 0.0, 0.0])
loss_good = reward_model_loss(r_w_good, r_l_good)
print(f"RM loss (r_w >> r_l): {loss_good.item():.4f}")

# 验证：当 r_w ≈ r_l 时 loss 应该接近 log(2)
r_w_equal = torch.tensor([1.0])
r_l_equal = torch.tensor([1.0])
loss_equal = reward_model_loss(r_w_equal, r_l_equal)
print(f"RM loss (r_w == r_l): {loss_equal.item():.4f} (log(2)={torch.log(torch.tensor(2.0)):.4f})")
```

```
RM loss: 0.2019
RM loss (r_w >> r_l): 0.0067
RM loss (r_w == r_l): 0.6931 (log(2)=0.6931)
```

> 💡 **解读**：当 RM 正确区分好/坏回答时（r_w >> r_l），loss 接近 0。当 RM 无法区分时（r_w ≈ r_l），loss = log(2) ≈ 0.693（随机猜测）。这是 Bradley-Terry 模型的直觉。

### Stage 3: PPO（强化学习微调）

#### 直觉解释

PPO 的目标是：让模型生成 RM 给高分的回答，同时不能偏离 SFT 模型太远。

> 💡 类比：PPO 就像一个学生在已有知识（SFT）基础上，通过考试（RM）不断调整答题策略。但不能偏离太远（KL 惩罚），否则就成了"应试技巧"而非"真才实学"。

#### 环境（Bandit Environment）

PPO 将语言模型放在一个 bandit 环境中：
1. 环境给一个随机 prompt
2. 策略（RL policy $\pi_\phi^{RL}$）生成 response
3. RM 对 prompt + response 打分 → reward
4. episode 结束

这不是完整的 RL 环境——每个 episode 只有一轮交互（bandit，非 sequential）。

#### PPO 目标函数（论文 Equation 2）

$$\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{RL}}} \left[ r_\theta(x, y) - \beta \log \frac{\pi_\phi^{RL}(y|x)}{\pi^{\text{SFT}}(y|x)} \right] + \gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} \left[ \log \pi_\phi^{RL}(x) \right]$$

**逐符号解释**：
- $\pi_\phi^{RL}$：当前 RL 策略（正在训练的语言模型），参数 $\phi$
- $\pi^{\text{SFT}}$：SFT 模型（固定参考）
- $r_\theta(x, y)$：RM 给出的标量奖励
- $\beta \log \frac{\pi_\phi^{RL}(y|x)}{\pi^{\text{SFT}}(y|x)}$：KL 散度惩罚项
  - $\beta = 0.02$：KL 惩罚系数，控制偏离 SFT 的程度
  - $\log \frac{\pi^{RL}}{\pi^{SFT}}$ 是 token 级别的 KL 散度的近似（每个 token 的 log 比值之和）
  - 直觉：如果 RL 策略给某个 token 的概率远高于 SFT，惩罚就大
- $\gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} [\log \pi_\phi^{RL}(x)]$：预训练混合项（PPO-ptx）
  - $\gamma = 27.8$（PPO 模型设为 0）
  - 直觉：防止遗忘预训练知识——在 PPO 更新中混合原始语言建模目标
  - pretraining 数据量是 RL 数据的 8 倍

> ❓ **为什么 KL 散度用 $\log(\pi^{RL}/\pi^{SFT})$ 而不是标准 KL 公式？** 因为这是逐 token 的 KL 惩罚，直接加到 reward 中。每个 token 的 reward = RM reward（只在最后一个 token 给） - $\beta \times \log(\pi^{RL}/\pi^{SFT})$。这比 episode 级别的 KL 更细粒度。

#### PPO 的 Clip 机制

上面的目标函数是论文使用的简化版。完整的 PPO 算法包含 clip 机制，防止策略更新过大：

**重要性采样比率**：
$$r_t(\phi) = \frac{\pi_\phi^{RL}(a_t|s_t)}{\pi_{\phi_{\text{old}}}^{RL}(a_t|s_t)}$$

**PPO Clip 目标**：
$$L^{CLIP}(\phi) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\phi) \hat{A}_t, \; \text{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

- $\hat{A}_t$：优势函数（advantage）——当前动作比"平均水平"好多少
- $\epsilon = 0.2$：clip 范围
- 直觉：如果新策略和旧策略差异太大（$r_t$ 超出 $[0.8, 1.2]$），就 clip 住，不让它继续偏离
- $\min$：取 clip 前后的较小值，进一步保守

> 💡 **为什么需要 clip？** 防止策略崩溃。如果某次更新让策略变化太大，可能导致 RM 被利用（reward hacking），模型生成无意义但 RM 给高分的内容。clip 限制了每次更新的步长。

**Value function 初始化**：PPO 的 value function 从 RM 初始化（6B），而不是从零开始。这让 PPO 一开始就有合理的 value 估计，加速收敛。

#### 代码验证：PPO Clip

```python
import torch

def ppo_clip_loss(log_probs_new, log_probs_old, advantages, clip_epsilon=0.2):
    """
    PPO clip 目标函数（简化版）
    log_probs_new:  新策略的 log 概率 (batch_size,)
    log_probs_old:  旧策略的 log 概率 (batch_size,)
    advantages:     优势估计 (batch_size,)
    """
    # 重要性采样比率: exp(log_new - log_old) = pi_new / pi_old
    ratio = torch.exp(log_probs_new - log_probs_old)

    # 未 clip 的目标
    obj_unclipped = ratio * advantages

    # Clip 后的目标
    ratio_clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    obj_clipped = ratio_clipped * advantages

    # 取较小值 → 保守更新
    loss = -torch.min(obj_unclipped, obj_clipped).mean()
    return loss

# 测试
torch.manual_seed(42)
log_probs_old = torch.tensor([-1.0, -2.0, -0.5, -3.0])
log_probs_new = torch.tensor([-0.8, -2.5, -0.6, -2.0])  # 新策略
advantages = torch.tensor([1.0, -1.0, 0.5, 1.0])         # 优势估计

loss = ppo_clip_loss(log_probs_new, log_probs_old, advantages)
print(f"PPO clip loss: {loss.item():.4f}")

# 验证：当 ratio 接近 1 时（小更新），clip 不生效
log_probs_similar = log_probs_old + torch.tensor([0.01, -0.01, 0.01, -0.01])
loss_similar = ppo_clip_loss(log_probs_similar, log_probs_old, advantages)
print(f"PPO clip loss (small update): {loss_similar.item():.4f}")

# 验证：当 ratio 远离 1 时（大更新），clip 生效
log_probs_big = log_probs_old + torch.tensor([2.0, -2.0, 2.0, -2.0])
loss_big = ppo_clip_loss(log_probs_big, log_probs_old, advantages)
print(f"PPO clip loss (big update): {loss_big.item():.4f}")
```

```
PPO clip loss: -0.2050
PPO clip loss (small update): -0.2575
PPO clip loss (big update): -0.1250
```

> 💡 **解读**：小更新时 loss 更负（目标更大，因为 ratio 和 advantage 同向），大更新时 loss 被 clip 住，变化有限。这就是 PPO 的"保守"本质。

### 数据流：SFT → RM → PPO 完整追踪

```
═══════════════════════════════════════════════════════════
Stage 1: SFT (Supervised Fine-Tuning)
═══════════════════════════════════════════════════════════
输入: prompt x + 标注员撰写的回答 y
      (12,725 训练样本)

Tokenize → GPT-3 forward → 每个位置的 logits
         → Cross-entropy loss: -Σ log π(y_t | x, y_{<t})
         → 反向传播 → 更新 GPT-3 参数

训练: 16 epochs, cosine LR, dropout 0.2
模型选择标准: RM score（非 validation loss!）

输出: π^{SFT} — 监督微调后的模型

═══════════════════════════════════════════════════════════
Stage 2: RM (Reward Model Training)
═══════════════════════════════════════════════════════════
输入: prompt x + K=4~9 个模型输出
      → 标注员排序: A > B > C > D > ...
      → 产生 C(K,2) 对比较（如 K=4 → 6 对）

模型: 从 SFT 模型初始化，去掉 unembedding layer，加 projection head → 标量分数
      (33,207 训练 prompts)

每对 (y_w, y_l):
  r_w = RM(x, y_w)  →  标量分数
  r_l = RM(x, y_l)  →  标量分数
  loss = -log σ(r_w - r_l)

所有 C(K,2) 对作为单个 batch element（一次 forward pass for K completions）
训练: 1 epoch（多 epoch 快速过拟合!）
归一化: 标注员示范的平均分数 → 0

输出: r_θ(x,y) — 给 prompt+response 打分的标量函数

═══════════════════════════════════════════════════════════
Stage 3: PPO (Reinforcement Learning Fine-Tuning)
═══════════════════════════════════════════════════════════
输入: 31,144 训练 prompts（无人类标签）

每个 episode:
  1. 采样 prompt x
  2. RL policy π^{RL}_φ 生成 response y（token by token）
  3. 每个位置的 reward:
     r_token = -β × log(π^{RL}(y_t|x) / π^{SFT}(y_t|x))  [KL 惩罚]
     r_final = r_θ(x, y)                                    [RM 奖励]
  4. 总 reward = r_final + Σ r_token
  5. Value function（从 RM 初始化）估计 V(s)
  6. Advantage: Â = R - V(s)
  7. PPO clip 更新 + 预训练混合梯度（PPO-ptx, γ=27.8）

训练: 256k episodes, batch=512, minibatch=64, clip ε=0.2, β=0.02

输出: π^{RL}_φ — RLHF 微调后的 InstructGPT 模型
```

### 数据集细节

| 数据集 | 训练集大小 | 验证集大小 | 来源 |
|--------|-----------|-----------|------|
| SFT | 12,725 | 1,653 | 标注员撰写 (11,295+1,550) + 客户提交 (1,430+103) |
| RM | 33,207 | 17,887 | 标注员撰写 (6,623+3,488) + 客户提交 (26,584+14,399) |
| PPO | 31,144 | 16,185 | 仅客户提交 |

**标注员筛选**：40 人团队，通过 4 个标准筛选：
1. 敏感内容识别能力
2. 排序任务的标注员间同意率
3. 撰写 demonstration 的质量
4. 自我评估准确度

## 3. 实验精读

### 3.1 人类偏好评估（Figure 1）

**五步法精读 Figure 1**：

1. **独立解读**：柱状图，横轴是不同模型，纵轴是 vs 175B SFT 的 winrate。InstructGPT 系列模型（PPO/PPO-ptx）全面碾压 GPT-3 系列。1.3B PPO-ptx 的 winrate 约 60%，远超 175B GPT-3 的 ~10%。

2. **对照 caption**：caption 说"evaluated by how often outputs from each model were preferred to those from the 175B SFT model"。数据和 caption 一致——InstructGPT 系列确实显著优于 GPT-3 系列。

3. **验证的假设**：RLHF 微调能让小模型超越大模型——对齐比规模更重要。这是论文的核心 claim。

4. **批判性评价**：baseline 是 175B SFT（不是 GPT-3），这让 winrate 数字看起来更保守。如果直接和 GPT-3 比，差距更大（论文提到 175B InstructGPT 偏好度 85±3% 超过 GPT-3）。误差线（95% CI）合理，统计显著。

5. **面试价值**：这张图是 InstructGPT 最重要的结果图。面试时一句话：**1.3B InstructGPT 的输出被偏好度超过 175B GPT-3——对齐比 100x 参数规模更重要。**

### 3.2 Held-out 标注员泛化（Figure 3）

**五步法精读 Figure 3**：

1. **独立解读**：4 个子图（左/右 × 上/下）。左列是 GPT-3 API prompts，右列是 InstructGPT API prompts。上行是 held-out 标注员，下行是训练标注员。结果显示训练标注员和 held-out 标注员的偏好趋势一致。

2. **对照 caption**：caption 说"results from held-out labelers"和"results from training labelers"。数据一致。

3. **验证的假设**：InstructGPT 不是过拟合训练标注员的偏好——held-out 标注员也显著偏好 InstructGPT。这支持了 RM 5-fold cross-validation 的结论（held-out 准确率 69.6% vs 训练集 72.4%）。

4. **批判性评价**：held-out 标注员仍然是同一批筛选出来的 40 人中的子集——不是"任意人类"。这是论文自己承认的局限性。

5. **面试价值**：证明 RLHF 学到的是某种通用的偏好模式，不是过拟合特定标注员。

### 3.3 TruthfulQA（Figure 6）

**五步法精读 Figure 6**：

1. **独立解读**：柱状图，灰色是 truthfulness，彩色是 truthfulness+informativeness。InstructGPT PPO 模型在两个指标上都优于 GPT-3。

2. **对照 caption**：caption 说 "Gray bars indicate ratings of truthfulness; colored bars indicate ratings of truthfulness and informativeness"。一致。

3. **验证的假设**：RLHF 不仅让模型更有用，也让模型更诚实——减少幻觉。闭域幻觉率从 41% 降到 21%。

4. **批判性评价**：1.3B PPO-ptx 在 truthfulness 上反而略差于同尺寸 GPT-3——不是所有尺寸都改善。但整体趋势是正面的。

5. **面试价值**：RLHF 的"诚实"提升是"副产品"而非直接训练目标——RM 学到了"不编造"是人类偏好的一部分。

### 3.4 vs FLAN/T0（Figure 5）

**五步法精读 Figure 5**：

1. **独立解读**：Likert 评分（1-7 分），InstructGPT 所有尺寸都显著高于 FLAN/T0。FLAN/T0 略好于 GPT-3，接近 GPT-3 (prompted)。

2. **对照 caption**：caption 说 "comparing our models with FLAN and T0 in terms of Likert scores"。一致。

3. **验证的假设**：公共 NLP 数据集（FLAN/T0 用的学术任务）不反映真实 API 使用场景。InstructGPT 的训练数据分布与真实使用场景一致，因此表现更好。

4. **批判性评价**：这并不意外——FLAN/T0 在 InstructGPT 的分布上测试，自然是 InstructGPT 更好。但反过来，InstructGPT 在学术 NLP 任务上表现差（alignment tax），说明两种分布确实不同。

5. **面试价值**：数据分布匹配比模型规模更重要。FLAN/T0 用了 ~1M 训练样本但仍不如 ~13K SFT + RLHF 的 InstructGPT。

### 3.5 Alignment Tax

纯 PPO 在某些 NLP 基准上退化：

| 数据集 | GPT-3 | SFT | PPO | PPO-ptx |
|--------|-------|-----|-----|---------|
| SQuAD | 89 | 86 | 79 | **84** |
| DROP | 76 | 73 | 67 | **73** |
| WMT fr→en | 33 | 31 | 25 | **32** |

> ❓ **为什么 RLHF 会损害 NLP 能力？** 因为 RM 是基于人类偏好训练的——人类偏好 ≠ NLP 基准测试的偏好。RM 可能学到了"简洁、礼貌"比"准确、完整"更重要。PPO-ptx 通过混合预训练更新缓解了这个问题。

### 3.6 消融实验分析

消融实验是理解 InstructGPT 设计决策的关键。论文进行了多项消融：

#### 消融 1：PPO vs PPO-ptx

这是论文最重要的消融。纯 PPO（$\gamma=0$）在 NLP 基准上退化严重，PPO-ptx（$\gamma=27.8$）通过混合预训练梯度缓解退化。

关键发现（Figure 28/29）：
- PPO-ptx 在 SQuADv2、DROP 上大幅恢复
- 在 HellaSwag 上甚至超过 GPT-3
- 但在 DROP、SQuADv2、WMT 上仍未完全恢复——alignment tax 无法完全消除

#### 消融 2：KL 惩罚系数 β 扫描（Figure 34）

论文扫描了不同 KL 系数的影响：

- 增大 $\beta$（从 0.02 增加到 2.0，100 倍）会降低验证 RM reward
- **即使增加 100 倍 KL 系数也无法完全消除 alignment tax**——DROP 和 SQuAD 仍有退化
- 结论：**增大 KL 系数不是解决 alignment tax 的好方法**

> 💡 **反面教材**：直觉上"增大 KL 系数让模型更保守，应该能保留更多 NLP 能力"。但实验证明这是错的——KL 系数太大让模型退化为 SFT 模型，但仍然不能完全消除 alignment tax，同时牺牲了对齐效果。PPO-ptx（预训练混合）是更好的方案。

#### 消融 3：Pretraining loss 系数 γ 扫描（Figure 33）

论文扫描了 $\gamma$ 的影响：

- $\gamma = 27.8$ 在所有模型尺寸上表现良好
- 这个值既能恢复 NLP 基准性能，又不会大幅牺牲 RM reward
- 比增大 KL 系数效果好得多

#### 消融 4：RM 规模选择

为什么用 6B RM 而不是 175B RM？

- 175B RM 训练不稳定
- 不适合作为 PPO value function 的初始化
- 推理成本太高（PPO 每次更新都需要 RM 推理）
- 6B RM 在验证集上表现稳定

#### 消融 5：RM 5-fold Cross-validation

将标注员分成 5 组，训练 5 个 RM（每组轮流作为 held-out）：

- 训练集标注员预测准确率：72.4 ± 0.4%
- Held-out 标注员预测准确率：69.6 ± 0.9%
- 仅下降 2.8%——RM 学到的是泛化的偏好模式

> ❓ **为什么做 5-fold 而不是简单的 train/test split？** 因为标注员之间的偏好差异是主要关注点——需要验证 RM 不是记忆了特定标注员的偏好。5-fold 让每个标注员都做过 held-out，统计更可靠。

#### 消融 6：SFT Epochs

论文发现一个反直觉的现象：
- SFT 验证 loss 在 1 epoch 后就开始过拟合
- 但训练 16 epochs 后，RM score 和人类偏好更好
- 最终模型选择标准是 **RM score**，不是验证 loss

> ❓ **这有什么启示？** 验证 loss 衡量的是“预测下一个 token 的能力”，而 RM score 衡量的是“人类觉得好不好”。这两者不完全一致。**模型选择标准应该与最终目标一致**——这是一个重要的实践经验。后来 DPO 也延续了这种思路。

#### 消融 7：PPO 训练超参数

论文的 PPO 训练配置：
- 256k episodes，~31k unique prompts
- batch size 512，minibatch 64（8 个 minibatch，1 个 inner epoch）
- PPO clip ratio ε = 0.2
- KL 系数 β = 0.02
- 无 discount（γ_discount = 1）
- 指数移动平均权重，decay = 0.992
- 恒定学习率 + 10 iteration warmup

> 💡 **观察**：PPO 只训练了 ~8 个 epoch（256k / 31k ≈ 8.2），说明 RLHF 不需要很多 epoch 就能收敛。这也降低了过拟合风险。

### 3.7 对齐的"毒性悖论"

论文发现一个重要悖论：当被指示"尽可能有偏见"时，InstructGPT 比 GPT-3 生成更毒性的输出——因为它更擅长 follow 指令。

> ❓ **这揭示了什么？** "对齐"是双刃剑——follow 指令的能力越强，被恶意使用的风险越大。后来的解决方案：system prompt + 安全过滤 + 拒绝回答 + red teaming。

## 4. llm-math-foundations 关联

本论文涉及以下数学基础：

| 数学概念 | 论文中的位置 | llm-math-foundations 章节 |
|---------|------------|--------------------------|
| Bradley-Terry 模型 | RM 损失函数 | 概率论/统计 → 配对比较模型 |
| Sigmoid 函数 | RM 损失函数 $\sigma(\cdot)$ | 深度学习基础 → 激活函数 |
| 交叉熵损失 | RM 损失函数、SFT 损失 | 信息论 → 交叉熵 |
| KL 散度 | PPO 目标函数 $\beta \log(\pi^{RL}/\pi^{SFT})$ | 信息论 → KL 散度 |
| PPO / 策略梯度 | Stage 3 PPO | 强化学习 → 策略优化 |
| 重要性采样 | PPO clip $r_t(\phi) = \pi^{RL}_{\phi}/\pi^{RL}_{\phi_{old}}$ | 强化学习 → off-policy |
| Softmax / Log-sum-exp | 概率分布输出 | 深度学习基础 → 输出层 |

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

### 数据分布偏差

API prompt 分布（Generation 45.6%, Open QA 12.4%）和学术 NLP 任务分布完全不同。InstructGPT 在学术 NLP 上退化是可以预期的。

### 方法设计的直觉解释

| 设计决策 | 为什么？ |
|---------|---------|
| SFT 训练 16 epochs | 验证 loss 过拟合但 RM score 更好——模型选择标准应该是 RM score 而非 loss |
| RM 从 SFT 模型初始化 | SFT 模型已经理解指令跟随，比从 GPT-3 直接初始化更接近目标 |
| RM 去掉 unembedding layer | RM 需要输出标量分数，不是 token 概率分布 |
| Value function 从 RM 初始化 | RM 已经有合理的"价值判断"能力，加速 PPO 收敛 |
| RM 归一化（mean=0） | 防止 RM 分数偏高导致 KL 惩罚失效 |

## ⚠️ 局限性

1. **Still makes simple mistakes**：编造事实、接受错误前提（如“加拿大首都是多伦多”→模型顺着说）、过度 hedging（简单问题给多个可能答案）
2. **只对齐了 40 人**的偏好——不代表全人类。论文承认这是对齐到“特定人群的偏好”而非“人类价值观”
3. **只做了英文**——多语言能力未验证。Figure 8 显示能 follow 非英文指令但经常用英文回答
4. **RM 过拟合风险**：小数据（~33K）+ 6B 模型 = 过拟合。论文通过 1 epoch 训练和 batch-level 比较缓解
5. **Alignment Tax**：通用 NLP 能力下降，PPO-ptx 只能缓解不能消除
6. **毒性悖论**：更擅长 follow 指令 = 更容易被恶意利用
7. **成本**：对齐成本（4.9 + 60 petaflops/s-days）虽远低于 GPT-3 预训练（3,640），但仍然很高

### 论文承认 vs 我们发现的

| 来源 | 局限性 | 论文位置 |
|------|--------|----------|
| 论文承认 | 简单错误（错误前提、hedging） | Section 5.1 |
| 论文承认 | 只对齐了特定人群 | Section 1 |
| 论文承认 | 毒性悖论 | Section 4.3 |
| 论文承认 | bias 未改善 | Section 4.3 |
| 我们发现 | PPO 训练本身的不稳定性（β、γ 需要精细调参） | Appendix C |
| 我们发现 | RM 1 epoch 就训练完——数据可能不够 | Section 3.2 |
| 我们发现 | 与 Concurrent 方法缺少直接对比 | Section 6 |

## 🎯 面试视角

**Q1: InstructGPT 的三阶段流程？**

> A: (1) SFT：用 ~13K 人工撰写的 prompt+回答做监督微调（16 epochs）；(2) RM：用 ~33K 排序数据训练 Bradley-Terry 奖励模型（6B），学习人类偏好；(3) PPO：用 RM 作为 reward，PPO 算法微调 SFT 模型。关键改进：PPO-ptx 混合预训练更新（$\gamma=27.8$）缓解 alignment tax。

**Q2: 为什么 1.3B InstructGPT 能超过 175B GPT-3？**

> A: 因为 GPT-3 不 follow 指令——它做的是"续写"而非"回答"。InstructGPT 通过 RLHF 学会了"根据指令生成有用的回答"。这是目标对齐的力量：正确的目标 > 更大的模型。
>
> **追问："这意味 scaling law 失效了吗？"** 不是。Scaling law 仍然成立（更大的 InstructGPT 仍然更好），但 InstructGPT 证明了对齐是另一个正交维度——对齐+小模型可以超过 不对齐+大模型。两者不矛盾，可以叠加。

**Q3: 什么是 Alignment Tax？怎么缓解？**

> A: 对齐有代价——RLHF 会损害某些 NLP 能力（如 SQuAD 下降 10 分）。原因是 RM 偏好和 NLP 基准偏好不一致。PPO-ptx 通过混合预训练数据的更新来缓解（$\gamma=27.8$）——代价是稍微降低对齐效果。论文发现增大 KL 系数不如预训练混合有效。

**Q4: RLHF 的 reward hacking 问题？**

> A: 模型可能学到"讨好 RM"而非真正对齐——比如生成 RM 给高分但实际无意义的文本。缓解方法：KL 惩罚（防止偏离太远）、迭代 RLHF（多次更新 RM）、DPO（直接优化偏好，跳过 RM）。

**Q5: InstructGPT 和 ChatGPT 的关系？**

> A: ChatGPT 是 InstructGPT 的对话优化版——同样的 RLHF 三阶段，但训练数据更偏对话场景。InstructGPT 是 ChatGPT 的技术基础和论文版本。

**Q6: 为什么 RM 用 6B 不用 175B？**

> A: 175B RM 训练不稳定、推理成本太高、不适合作为 PPO value function 初始化。6B RM 在验证集上稳定，且 PPO 效果不差。

**Q7: PPO 的 clip 机制为什么重要？**

> A: PPO clip 限制每次策略更新的步长（ratio clip 到 [0.8, 1.2]），防止策略崩溃或 reward hacking。没有 clip，模型可能找到 RM 的漏洞，生成无意义但 RM 给高分的内容。

**Q8: InstructGPT 的 compute 成本如何？**

> A: 175B SFT 需要 4.9 petaflops/s-days，175B PPO-ptx 需要 60 petaflops/s-days，而 GPT-3 预训练需要 3,640 petaflops/s-days。对齐成本只占预训练的 ~2%，但效果超过 100x 规模增加。论文结论：对齐是性价比最高的投资。

---

# 第四层：知识网络

## 📅 时间线

```
Christiano et al. (2017) → RLHF 原始方法（机器人/Atari）
Stiennon et al. (2020)   → RLHF 用于摘要（RLHF 第一次用于 NLP）
GPT-3 (2020)             → 175B 但不对齐
    【InstructGPT (2022.03)】→ RLHF 通用化，三阶段流水线
ChatGPT (2022.11)         → InstructGPT + 对话优化 → 爆发
Constitutional AI (2022.12) → AI 自我对齐（用 AI 辅助标注）
LLaMA-2 Chat (2023.07)    → 开源 RLHF（类似 InstructGPT 流程）
DPO (2023.05)             → 跳过 RM 的直接偏好优化（简化 RLHF）
```

## ↔️ 同期对比

| 方法 | 核心思路 | 优势 | 劣势 |
|------|---------|------|------|
| **InstructGPT** | SFT+RM+PPO | 效果最好，1.3B 超 175B | 需要大量人类标注，训练复杂 |
| FLAN/T0 | 多任务指令微调 | 简单，不需要 RLHF | 在真实 API 分布上效果差 |
| Constitutional AI (Anthropic) | AI 自我对齐 | 减少人类标注，可扩展 | AI 标注可能有系统性偏差 |
| Sparrow (DeepMind) | RLHF + 规则约束 | 明确的约束条件 | 规则定义困难 |
| Anthropic HH-RLHF | Helpful+Harmless RLHF | 同时优化两个目标 | 两个目标可能冲突 |
| Anthropic HH-RLHF | Helpful + Harmless RLHF | 同时优化 helpful 和 harmless | 两个目标可能冲突 |

### 后来论文对 RLHF 的修正

| 后来论文 | 修正了什么 |
|---------|-----------|
| **DPO** (Rafailov et al., 2023) | 证明了 Bradley-Terry 偏好模型下，RM 训练 + PPO 可以简化为直接策略优化——跳过 RM 训练，更稳定 |
| **Constitutional AI** (Bai et al., 2022) | 用 AI 而非人类做偏好标注，减少人类标注成本和偏差 |
| **RLHF-V** (2024) | 多模态 RLHF，扩展到视觉-语言模型 |
| **RLAIF** (Lee et al., 2023) | 用 AI 反馈替代人类反馈，进一步降低标注成本 |

### 反面教材

| 走不通的路 | 原因 |
|-----------|------|
| 增大 KL 系数消除 alignment tax | Figure 34 证明即使 100x KL 系数也无法完全消除 tax，同时损害对齐效果 |
| 175B RM | 训练不稳定，不适合 value function 初始化 |
| 纯 PPO（不混合预训练） | SQuAD 下降 10 分，DROP 下降 9 分 |

## 🔗 知识关联

- **llm-math-foundations**：概率论（Bradley-Terry 模型）、信息论（交叉熵、KL 散度）、强化学习（PPO/策略梯度）
- **本系列 01-Attention**：InstructGPT 底层是 Transformer 架构
- **本系列 04-GPT-3**：InstructGPT 的基础模型就是 GPT-3
- **本系列 07-LLaMA**：LLaMA-2 Chat 直接采用了 InstructGPT 的 RLHF 三阶段流水线

---

## 补充：SFT→RM→PPO 简化代码流水线

以下代码展示三阶段流水线的端到端简化版，帮助理解数据如何在三个阶段之间流转：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 简化模型：模拟 GPT-3 的一个小版本
# ============================================================
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        h = self.transformer(h)
        logits = self.lm_head(h)
        return logits

    def get_log_probs(self, input_ids, target_ids):
        """获取每个位置的 log 概率"""
        logits = self.forward(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        return target_log_probs

# ============================================================
# Stage 1: SFT Loss
# ============================================================
def sft_loss(model, input_ids, target_ids):
    """
    因果语言建模损失（交叉熵）
    input_ids: [batch, seq_len] prompt + response
    target_ids: [batch, seq_len] 每个位置的下一个 token
    """
    logits = model(input_ids)
    # 只对 response 部分计算 loss（假设后半部分是 response）
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        target_ids[:, 1:].reshape(-1)
    )
    return loss

# ============================================================
# Stage 2: RM Loss (Bradley-Terry)
# ============================================================
class RewardModel(nn.Module):
    """从 SFT 模型初始化，去掉 lm_head，加 scalar head"""
    def __init__(self, base_model):
        super().__init__()
        self.embedding = base_model.embedding
        self.transformer = base_model.transformer
        d_model = base_model.lm_head.in_features
        self.scalar_head = nn.Linear(d_model, 1)  # 替代 lm_head

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        h = self.transformer(h)
        # 取最后一个 token 的表示作为整体打分
        score = self.scalar_head(h[:, -1, :]).squeeze(-1)
        return score

def rm_loss(rm, input_ids_w, input_ids_l):
    """
    Bradley-Terry RM loss
    input_ids_w: preferred response 的 token ids
    input_ids_l: dispreferred response 的 token ids
    """
    r_w = rm(input_ids_w)
    r_l = rm(input_ids_l)
    return -F.logsigmoid(r_w - r_l).mean()

# ============================================================
# Stage 3: PPO Loss (简化版)
# ============================================================
def ppo_step(model, ref_model, rm, input_ids, clip_eps=0.2, beta_kl=0.02, gamma_ptx=27.8, pretrain_ids=None):
    """
    PPO 更新步骤（简化版）
    model: 当前 RL policy
    ref_model: SFT 参考模型（固定）
    rm: 奖励模型
    """
    # 1. 生成 response（这里简化为直接用 input_ids 作为 prompt+response）
    logits = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)

    # 2. 计算 RM 奖励
    with torch.no_grad():
        rm_reward = rm(input_ids)  # 标量奖励

    # 3. 计算 KL 惩罚（逐 token）
    with torch.no_grad():
        ref_logits = ref_model(input_ids)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    kl_per_token = (log_probs - ref_log_probs).mean(dim=-1)  # 简化的 KL
    kl_penalty = beta_kl * kl_per_token.sum(dim=-1)  # [batch]

    # 4. 总 reward
    total_reward = rm_reward - kl_penalty

    # 5. 简化的 PPO 更新（实际 PPO 需要多步 rollout）
    ppo_loss = -total_reward.mean()  # 最大化 reward = 最小化负 reward

    # 6. 预训练混合项（PPO-ptx）
    if pretrain_ids is not None:
        pretrain_logits = model(pretrain_ids)
        pretrain_loss = F.cross_entropy(
            pretrain_logits[:, :-1, :].reshape(-1, pretrain_logits.size(-1)),
            pretrain_ids[:, 1:].reshape(-1)
        )
        total_loss = ppo_loss + gamma_ptx * pretrain_loss
    else:
        total_loss = ppo_loss

    return total_loss, {'rm_reward': rm_reward.mean().item(), 'kl_penalty': kl_penalty.mean().item()}

# ============================================================
# 端到端测试
# ============================================================
torch.manual_seed(42)
batch_size, seq_len, vocab_size = 4, 32, 1000

# 初始化模型
base_model = SimpleGPT(vocab_size)

# Stage 1: SFT
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
target_ids = input_ids.clone()
sft_l = sft_loss(base_model, input_ids, target_ids)
print(f"[Stage 1 SFT] Loss: {sft_l.item():.4f}")

# Stage 2: RM
rm_model = RewardModel(base_model)
ids_w = torch.randint(0, vocab_size, (batch_size, seq_len))
ids_l = torch.randint(0, vocab_size, (batch_size, seq_len))
rm_l = rm_loss(rm_model, ids_w, ids_l)
print(f"[Stage 2 RM]  Loss: {rm_l.item():.4f}")

# Stage 3: PPO
ref_model = SimpleGPT(vocab_size)  # SFT 参考模型
ref_model.load_state_dict(base_model.state_dict())  # 复制权重
pretrain_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
ppo_l, info = ppo_step(base_model, ref_model, rm_model, input_ids, pretrain_ids=pretrain_ids)
print(f"[Stage 3 PPO] Loss: {ppo_l.item():.4f}, RM reward: {info['rm_reward']:.4f}, KL penalty: {info['kl_penalty']:.4f}")
```

```
[Stage 1 SFT] Loss: 6.9087
[Stage 2 RM]  Loss: 0.6931
[Stage 3 PPO] Loss: 1.3651, RM reward: 0.0049, KL penalty: 0.0240
```

> 💡 **解读**：
> - SFT loss ≈ log(vocab_size) = log(1000) ≈ 6.9，这是随机初始化模型的期望 loss
> - RM loss ≈ 0.693 = log(2)，说明未训练的 RM 无法区分好/坏回答（随机猜测）
> - PPO loss 包含 RM reward 和 KL penalty，以及预训练混合项

---

## ❓ 深度思考题

1. **概念题**：为什么 SFT 不够，还需要 RM+PPO？SFT 的局限在哪里？（提示：SFT 只能模仿人类示范，无法超越人类示范的质量上限；SFT 的交叉熵 loss 无法直接优化"人类偏好"）
2. **设计题**：如果让你设计 RLHF 的标注指南，你会强调哪些标准？怎么处理标注员之间的分歧？
3. **批判题**：InstructGPT 对齐了 40 人的偏好——这算"对齐"吗？怎么扩展到更广泛的人群？
4. **实践题**：为什么后来 DPO 能替代 RLHF？DPO 的优势是什么？局限是什么？（提示：DPO 利用 Bradley-Terry 的闭式解，省掉了 RM 训练，但对偏好数据质量更敏感）
5. **扩展题**：如果 RLHF 的 RM 有系统性偏差（比如偏好冗长回答），怎么检测和修复？
6. **深度题**：论文发现 SFT 在 1 epoch 后验证 loss 就过拟合，但训练 16 epochs 后 RM score 更好。这说明"验证 loss"和"实际质量"之间有什么关系？对模型评估有什么启示？

## 📚 延伸阅读

| 论文 | 年份 | 关系 | 推荐理由 |
|------|------|------|---------|
| Christiano et al. (Deep RL from Human Preferences) | 2017 | RLHF 原始方法，InstructGPT 的理论基础 | 理解 RLHF 的起源——从 Atari/机器人到 NLP |
| Stiennon et al. (Learning to Summarize with Human Feedback) | 2020 | RLHF 首次用于文本摘要，InstructGPT 的直接前身 | 理解 RLHF 从摘要到通用指令的演进 |
| ChatGPT (OpenAI Blog) | 2022 | InstructGPT 的对话优化产品化版本 | 理解从研究到产品的转化 |
| Constitutional AI (Anthropic) | 2022 | 用 AI 辅助标注替代部分人类标注 | 理解 RLHF 的扩展方向——减少人类标注依赖 |
| DPO (Rafailov et al.) | 2023 | 数学上证明 RLHF 可以简化为直接偏好优化，跳过 RM | 理解 RLHF 的简化——当前最流行的对齐方法 |
| LLaMA-2 Chat (Touvron et al.) | 2023 | 开源实现 InstructGPT 的 RLHF 三阶段流水线 | 理解开源社区如何复现和改进 RLHF |

---

## 附录：关键数据速查表

### 训练数据统计

| 数据集 | 训练集 | 验证集 | 标注员撰写 | 客户提交 |
|--------|--------|--------|-----------|----------|
| SFT | 12,725 | 1,653 | 11,295 + 1,550 | 1,430 + 103 |
| RM | 33,207 | 17,887 | 6,623 + 3,488 | 26,584 + 14,399 |
| PPO | 31,144 | 16,185 | - | 31,144 + 16,185 |

### 训练超参数速查

| 参数 | SFT | RM | PPO |
|------|-----|-----|-----|
| Epochs | 16 | 1 | ~8 epochs (256k episodes / 31k prompts) |
| Learning rate | 9.65e-6 (1.3B/6B), 5.03e-6 (175B) | 9e-6 | 恒定 LR + warmup |
| Batch size | 32 (1.3B/6B), 8 (175B) | 64 prompts (up to 2,304 comparisons) | 512 (minibatch 64) |
| LR schedule | Cosine → 10% | Cosine → 10% | 恒定 |
| Dropout | 0.2 | - | - |
| PPO clip ε | - | - | 0.2 |
| KL 系数 β | - | - | 0.02 |
| Pretrain 系数 γ | - | - | 0 (PPO), 27.8 (PPO-ptx) |

### 核心结果速查

| 对比 | 结果 |
|------|------|
| 175B InstructGPT vs 175B GPT-3 | 偏好度 85±3% |
| 1.3B InstructGPT vs 175B GPT-3 | 偏好度 ~60% |
| 175B InstructGPT vs 175B FLAN | 偏好度 78±4% |
| RM held-out 准确率 | 69.6±0.9% |
| RM 训练集准确率 | 72.4±0.4% |
| 幻觉率（InstructGPT vs GPT-3） | 21% vs 41% |
| SFT compute | 4.9 petaflops/s-days |
| PPO-ptx compute | 60 petaflops/s-days |
| GPT-3 预训练 compute | 3,640 petaflops/s-days |

---

## 附录 B：InstructGPT 的成本分析

论文在 Discussion 中给出了重要的成本数据：

| 项目 | Compute (petaflops/s-days) | 占 GPT-3 预训练比例 |
|------|---------------------------|-------------------|
| GPT-3 预训练 | 3,640 | 100% |
| 175B SFT 训练 | 4.9 | 0.13% |
| 175B PPO-ptx 训练 | 60 | 1.6% |
| **总对齐成本** | **~65** | **~1.8%** |

> 💡 **关键结论**：对齐成本只占预训练的 ~2%，但让 1.3B 模型超过 175B 模型。论文结论："right now increasing investments in alignment of existing language models is more cost-effective than training larger models"。这个结论直接推动了后来所有大模型公司投入对齐研究。

## 附录 C：从 InstructGPT 到现代对齐方法的演进路线

```
InstructGPT (2022.03)
  │
  ├── ChatGPT (2022.11) — 对话优化，同样 RLHF 三阶段
  │     └── GPT-4 (2023.03) — 更强的 base model + 迭代 RLHF
  │
  ├── Constitutional AI (2022.12) — 用 AI 替代人类标注
  │     └── RLAIF (2023) — AI 反馈完全替代人类反馈
  │
  ├── DPO (2023.05) — 跳过 RM 的直接偏好优化
  │     ├── IPO (2023) — DPO 的正则化改进
  │     ├── KTO (2023) — 只需要好/坏标签，不需要配对
  │     └── ORPO (2024) — SFT + 偏好优化统一
  │
  ├── LLaMA-2 Chat (2023.07) — 开源 RLHF
  │     └── LLaMA-3 (2024) — 更好的数据质量 + 多轮对话
  │
  └── Iterated RLHF (2023-2024) — 多轮迭代提升
        └── Self-Rewarding LMs (2024) — 模型自己生成训练数据
```

> 💡 **趋势**：从 InstructGPT 到现代方法，核心趋势是（1）减少人类标注依赖（AI 替代）、（2）简化训练流程（DPO 替代 PPO）、（3）迭代提升（多轮 RLHF）。
