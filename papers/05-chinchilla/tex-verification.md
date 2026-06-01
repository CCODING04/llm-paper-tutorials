# TeX 源码验证记录

## 论文信息
- arXiv ID: 2203.15556
- 验证日期: 2026-06-01

## 验证方法
下载 arXiv LaTeX 源码，对比 tex caption/表格数据与 raw-extract.md 中的内容。

## 发现的问题
1. raw-extract.md 中部分公式符号丢失（如 N 和 D 变为 ??）——MinerU 转换问题
2. Table 2 的数值在 raw-extract.md 中与 tex 源码一致（a=0.50, 0.49, 0.46 vs Kaplan 0.73）
3. MMLU 结果 tex 源码：Random 25.0%, GPT-3 43.9%, Gopher 60.0%, Chinchilla 67.6%, Human expert 89.8% ✓

## 修复内容
- 在 merged-tutorial.md 和 figures-analysis.md 中使用 tex 源码的精确数值
- 公式符号以 tex 源码为准
