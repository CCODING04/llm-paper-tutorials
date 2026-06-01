# InstructGPT 论文图表分析

---

## Figure 1: 人类偏好评估

### 客观描述（5A）
柱状图，展示各模型相对于 175B SFT 模型的偏好率：
- 1.3B PPO-ptx（InstructGPT）：~85%（vs 175B GPT-3）
- 175B PPO-ptx：~85%（vs 175B GPT-3）
- 175B GPT-3 few-shot：显著低于 InstructGPT
- 误差棒为 95% 置信区间

### 深度分析（5B）
- **核心发现**：1.3B InstructGPT 偏好率 ≈ 175B InstructGPT——对齐效果与规模无关
- **验证的假设**：直接证明"对齐比规模更重要"
- **批判**：评估只在 OpenAI API prompt 分布上——对其他类型任务可能不同
- **面试价值**：这张图是 InstructGPT 最核心的论据——1.3B > 175B

---

## Figure 2: RLHF 三步法流程图

### 客观描述（5A）
三步流程示意图：
1. SFT：prompt → 人类写演示 → 监督微调
2. RM：prompt → 多个模型输出 → 人类排序 → 训练奖励模型
3. PPO：prompt → SFT模型 → 输出 → RM评分 → PPO更新

### 深度分析（5B）
- **独立解读**：这是 RLHF 的标准流程图——定义了后来的对齐范式
- **面试价值**：面试必考——解释 RLHF 三步法

---

## Table 1: API Prompt 分布

### 客观描述（5A）
| 用途 | 比例 |
|------|------|
| Generation | 45.6% |
| Open QA | 12.4% |
| Brainstorming | 11.2% |
| Chat | 8.4% |
| Rewrite | 6.6% |
| Summarization | 4.2% |
| Classification | 3.5% |

### 深度分析（5B）
- **独立解读**：近半是生成类任务——这是 InstructGPT 最擅长的
- **批判**：Chat 只有 8.4%——后来的 ChatGPT 大幅增加了对话数据
- **面试价值**：说明对齐数据的分布决定了模型擅长的任务
