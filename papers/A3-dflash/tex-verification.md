# TeX 源码验证记录

## 论文信息
- arXiv ID: 2602.06036
- 论文标题: DFlash: Block Diffusion for Flash Speculative Decoding
- 验证日期: 2026-06-10

## 源码结构
```
main.tex → sections/{abstract,intro,related,preliminaries,method,exp,conclusion,appendix}.tex
```

## 发现的问题
- 无重大 typo
- 论文结构清晰，MinerU 转换的 raw-extract.md 与 LaTeX 源码一致

## 关键数值核对

| 数据点 | LaTeX 源码 | raw-extract.md | 一致性 |
|--------|-----------|----------------|--------|
| 最大加速 | 6.1× | 6.1× | ✅ |
| vs EAGLE-3 | 2.5× faster | 2.5× | ✅ |
| Draft 模型额外开销 | ~42MB (W_c) | 未明确提及 | ⚠️ raw-extract 未提取具体数值 |
| 现有方法上限 | 2-3× (EAGLE-3) | 2-3× | ✅ |
| DiffuSpec/SpecDiff-2 上限 | 3-4× | 3-4× | ✅ |

## 图片验证
- Figure 1-5 的 caption 与 raw-extract.md 中的描述一致
- 图片文件名：MinerU 自动 hash 命名，与 LaTeX 中的 `figures/dflash_speedup.pdf` 等不对应（正常）
- 5 个 Figure 的图片全部成功提取

## 修复内容
- 无需修复
