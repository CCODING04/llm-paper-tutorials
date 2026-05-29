# 🤖 论文教程自动化生成流程

本仓库中的每篇论文教程都遵循以下流程生成。添加新论文时，按步骤执行即可。

## 前置条件

- **MinerU CLI**：`npm install -g mineru-open-api`（v0.5.9+）
- **MINERU_TOKEN**：存储在 `~/.openclaw/secrets.json` 的 `MINERU_API_KEY`
- **OpenClaw**：用于调用 mimo-v2.5（图表分析）和 glm-5.1（教程生成）

## 流程步骤

### Step 1: 下载论文 PDF

```bash
# 从 arXiv 下载
curl -L -o paper.pdf "https://arxiv.org/pdf/XXXX.XXXXXvN"
# 或使用论文 DOI / 其他来源
```

### Step 2: MinerU 精细化转换

```bash
export MINERU_TOKEN=$(cat ~/.openclaw/secrets.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('MINERU_API_KEY',''))")

# 创建论文目录
mkdir -p papers/XX-paper-name/images

# 精细化提取（VLM 模式，保留图片）
mineru-open-api extract paper.pdf -o papers/XX-paper-name/ -f md --model vlm

# 重命名提取结果
mv papers/XX-paper-name/paper.md papers/XX-paper-name/raw-extract.md

# 图片已在 papers/XX-paper-name/images/ 中
```

### Step 3: 用 mimo-v2.5 分析论文图表

使用 mimo-v2.5 多模态模型逐个读取论文中的图表，生成详细的中文描述。

```
Prompt: 读取图片 [路径]，这是论文 [标题] 中的 Figure X。
请用中文详细描述这张图：
1. 图的整体结构（框图/曲线/表格等）
2. 每个组件的名称和作用
3. 数据/信息流向
4. 关键数值或标注
```

输出保存为 `figures-analysis.md`。

### Step 4: 用 glm-5.1 生成完整教程

使用教学 prompt 生成中文教程 `README.md`。

**教学 Prompt**：
```
You are the lead teacher of this classroom. You teach with clarity, warmth, and genuine enthusiasm for the subject matter.

Your teaching style:
- Explain concepts step by step, building from what students already know
- Use vivid analogies, real-world examples, and visual aids to make abstract ideas concrete
- Pause to check understanding — ask questions, not just lecture
- Adapt your pace: slow down for difficult parts, move briskly through familiar ground
- Encourage students by name when they contribute, and gently correct mistakes without embarrassment

Tone: Professional yet approachable. Patient. Encouraging. You genuinely care about whether students understand.
```

**教程结构要求**：
1. 论文背景与动机（为什么需要这篇论文？）
2. 核心创新点（一句话概括 + 展开讲解）
3. 逐节精读
   - 每个公式都要 step by step 推导，不跳步
   - 关联 llm-math-foundations 对应章节
   - 图表详细解读（使用 Step 3 的分析结果）
   - 辅助代码验证（关键公式用 Python 代码演示）
4. 关键实验结果解读
5. 思考题（3-5 题，巩固理解）
6. 延伸阅读

### Step 5: 复制原始 PDF 并推送到 GitHub

```bash
cp paper.pdf papers/XX-paper-name/
git add .
git commit -m "feat: add XX-paper-name tutorial"
git push origin main
```

## 质量检查清单

- [ ] MinerU 提取完整，图片引用正常
- [ ] 所有图表都有 mimo-v2.5 的中文分析
- [ ] 教程中公式推导不跳步
- [ ] 关联了 llm-math-foundations 的对应章节
- [ ] 包含辅助代码验证
- [ ] 思考题有针对性
- [ ] 整体读起来像老师在讲课，不像 AI 翻译

## 当前支持的论文

| # | 论文 | MinerU 模型 | 状态 |
|---|------|------------|------|
| 01 | Attention Is All You Need | VLM | ✅ 完成 |
