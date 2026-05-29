# 📚 LLM 论文精读教程

> 用 [llm-math-foundations](https://github.com/CCODING04/llm-math-foundations) 的数学基础，带你逐篇精读 LLM 领域的经典论文。

## 🎯 项目目标

1. **以数学为锚点**：每篇论文的学习都关联到 llm-math-foundations 中的数学知识
2. **Step by Step 教学**：不是论文翻译，而是真正的教学——有引导、有提问、有代码验证
3. **多维度阅读**：每篇论文提供独立讲解 + 论文原文融合讲解两种模式

## 📖 论文列表

| # | 论文 | 年份 | 核心主题 | 状态 |
|---|------|------|---------|------|
| 01 | [Attention Is All You Need](./papers/01-attention-is-all-you-need/) | 2017 | Transformer 架构 | ✅ 完成 |

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
