# InstructGPT 论文图表分析

> 数据来源：tex 源码 caption + raw-extract.md 精确数值

---

## Figure 1: 人类偏好评估（核心结果）

### 客观描述（5A）

柱状图，纵轴为 winrate vs 175B SFT baseline。
- 横轴：GPT / GPT (prompted) / SFT 1.3B / SFT 6B / SFT 175B / PPO 1.3B / PPO 6B / PPO 175B / PPO-ptx 1.3B / PPO-ptx 6B / PPO-ptx 175B
- 误差线：95% 置信区间

### 深度分析（5B）

- **独立解读**：PPO-ptx 175B 的 winrate 约 85%，远超 GPT-3 的 ~10%。更震惊的是 PPO-ptx 1.3B 的 winrate 约 60%，也远超 175B GPT-3。
- **对照 caption**："Human evaluations of various models on our API prompt distribution, evaluated by how often outputs from each model were preferred to those from the 175B SFT model." 与图一致。
- **验证的假设**：验证了 RLHF 对齐的有效性——不只是 175B，1.3B 也能大幅超越未对齐的 175B。
- **批判**：baseline 是 175B SFT 而非 GPT-3——这让所有 PPO 模型的 winrate 看起来更高（因为 SFT 本身就比 GPT-3 好）。如果 baseline 是 GPT-3，PPO 模型的 winrate 可能 >95%。
- **面试价值**：最核心的图——1.3B InstructGPT > 175B GPT-3 = "对齐比规模重要"。

---

## Figure 2: 三阶段流程图

### 客观描述（5A）

示意图，展示 SFT → RM → PPO 三阶段：
- Step 1: Prompt → 人工撰写回答 → SFT 训练
- Step 2: Prompt → 模型生成多个输出 → 标注员排序 A>B>C>D → RM 训练
- Step 3: Prompt → PPO 模型生成 → RM 打分 → PPO 更新

### 深度分析（5B）

- **独立解读**：清晰展示了 RLHF 的数据流——从人类演示到人类偏好到强化学习
- **面试价值**：面试必考——画图解释 RLHF 三阶段

---

## Figure 6: TruthfulQA 结果

### 客观描述（5A）

柱状图，灰色=truthfulness，彩色=truthfulness+informativeness
- InstructGPT truthful+informative 比例约 GPT-3 的 2 倍

### 深度分析（5B）

- **独立解读**：InstructGPT 在 truthfulness 上大幅改善——不仅更诚实，还保持了信息量
- **批判**：TruthfulQA 是对抗性设计的（专门针对 GPT-3 的弱点），非对抗性子集的效果如何？论文说同样有效。
- **面试价值**：证明 RLHF 不仅改善指令跟随，也改善了真实性

---

## Figure 8: RealToxicityPrompts

### 客观描述（5A）

散点图：人类评估毒性 vs Perspective API 自动评估毒性
- 1,729 prompts，3 个 175B 模型，有/无"respectful"指令

### 深度分析（5B）

- **独立解读**：InstructGPT 在加"respectful"指令后，毒性输出减少约 25%
- **批判**：改善不大，且不加指令时 InstructGPT 和 GPT-3 差不多。说明 RLHF 对毒性的改善有限。
- **面试价值**：证明 RLHF 不是万能的——对齐不能解决所有安全问题
