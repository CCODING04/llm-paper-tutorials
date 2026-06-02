# TeX 源码验证记录

## 论文信息
- arXiv ID: 2501.12948
- 验证日期: 2026-06-02

## 关键数值验证
- ✅ 基于 DeepSeek-V3-Base
- ✅ GRPO 强化学习框架（无需 critic 模型）
- ✅ 纯 RL 训练（R1-Zero），无需 SFT 前置
- ✅ 涌现行为：self-reflection、verification、动态策略调整
- ✅ 两阶段 RL + rejection sampling + SFT（R1 完整版）
- ✅ 蒸馏模型：1.5B/7B/14B/32B/70B 公开
- ✅ 超越 o1-preview 在多个 benchmark

## 发现的问题
- MinerU 提取中部分公式符号可能被 ? 替代
- 3 张图（AIME曲线、长度曲线、训练流水线）

## 修复内容
- 教程中使用正确数学符号
- 图表编号与 LaTeX 源码对齐
