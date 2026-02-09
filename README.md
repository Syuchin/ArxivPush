# 📚 ArXiv 每日论文推送助手

自动抓取 ArXiv 最新 AI 论文（通过 OpenAlex API），使用 LLM 进行深度分析，并推送到飞书。

## ✨ 功能特性

- 🔍 **自动抓取**：每日自动获取 ArXiv 最新 LLM Safety / Agent Safety / AI Agent 相关论文（使用 OpenAlex API，避免 GitHub Actions IP 限制）
- 🤖 **AI 深度分析**：调用智谱 GLM-4.7 API 生成结构化中文解读：
  - 【相关性】：1-5 分评估论文与研究领域的相关性
  - 【问题定义】：研究背景和核心瓶颈
  - 【方法核心】：关键技术思路和"为什么能 work"的直觉
  - 【主要发现】：实验结论和量级对比
  - 【局限性推测】：推测可能的局限性
  - 【潜在关联】：从 mechanistic interpretability、tokenization fragmentation、agent robustness 三个角度判断是否可为研究提供借鉴
  - 【一句话结论】：口语化总结，帮助快速判断是否值得精读
- 📊 **智能排序**：按相关性评分自动排序，优先推送高分论文
- 🎯 **智能过滤**：自动过滤低相关性论文（评分 < 3 分），只推送前 20 篇最相关的论文
- 💻 **代码链接**：自动从 PapersWithCode 匹配开源代码
- 📱 **飞书推送**：生成精美富文本卡片推送至飞书群

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置

**需要配置 3 个敏感信息**（其他配置已写死在代码中）：

```bash
export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/你的Webhook地址"
export DEEPSEEK_API_KEY="你的智谱API Key"
export OPENALEX_EMAIL="your-email@example.com"  # 用于 OpenAlex Polite Pool，提高请求优先级
```

你也可以把上述变量写到项目根目录的 `.env` 文件，脚本会自动加载。

- 飞书 Webhook：在飞书群设置 → 添加机器人 → 自定义机器人 → 获取 Webhook 地址
- 智谱 GLM API Key：在 [智谱开放平台](https://open.bigmodel.cn/) 获取
- OpenAlex Email：填写你的邮箱地址（任何有效邮箱即可），用于加入 OpenAlex Polite Pool，获得更高的请求优先级和速率限制

**默认配置**（已写死在代码中，无需修改）：
- 模型：`glm-4-7`（使用 Coding API 端点）
- 查询关键词：`LLM safety`, `agent safety`, `AI agent`, `language model safety`, `autonomous agent`
- 获取数量：50 篇（筛选后推送前 20 篇高分论文）
- 评分阈值：3.0 分（低于此分数的论文不推送）

### 3. 本地运行

```bash
python3 daily_paper.py
```

常用可选参数：

- `--max-results 3`：抓取论文数量
- `--query 'abs:LLM OR abs:\"AI Agent\"'`：自定义检索关键词
- `--prompt-file prompts/deepseek_summary_prompt.zh.j2`：自定义 Prompt 模板（Jinja2，默认已提供）
- `--per-paper` 或 `FEISHU_PER_PAPER=1`：每篇论文单独发一张卡片
- `--dry-run`：只打印，不推送飞书（调试用）
- `--skip-llm`：不调用 DeepSeek，仅输出摘要（用于无 Key 的本地测试）

### 4. 设置每日自动运行（GitHub Actions，推荐）

仓库已提供工作流：`.github/workflows/arxiv-daily.yml`，支持定时与手动触发。

#### 配置步骤：

1. **推送代码到 GitHub**
   ```bash
   git remote add origin git@github.com:你的用户名/你的仓库名.git
   git branch -M main
   git push -u origin main
   ```

2. **配置 GitHub Secrets**（必需）

   进入仓库页面 → Settings → Secrets and variables → Actions → New repository secret

   **需要添加 3 个 Secrets**：
   - `FEISHU_WEBHOOK`：你的飞书 Webhook 地址
   - `DEEPSEEK_API_KEY`：你的智谱 GLM API Key
   - `OPENALEX_EMAIL`：你的邮箱地址（用于 OpenAlex Polite Pool）

   其他配置（模型、查询关键词、推送数量等）已写死在代码中，无需配置。

3. **启用 GitHub Actions**

   进入仓库页面 → Actions → 启用工作流

   可以点击 "Run workflow" 手动触发一次测试

4. **定时运行**

   工作流默认每天 UTC 00:00（北京时间 08:00）自动运行

   如需修改时间，编辑 `.github/workflows/arxiv-daily.yml` 中的 `cron` 表达式

> 注意：GitHub Actions 的 cron 使用 UTC 时间；如需改运行时间，编辑 `.github/workflows/arxiv-daily.yml` 里的 `cron`。

### 5. 注意事项

- 确保网络可访问 OpenAlex API、DeepSeek API 和飞书服务器
- OpenAlex API 相比 ArXiv 官方 API 更稳定，特别适合 GitHub Actions 环境
- 建议先手动运行测试，确认配置无误后再设置定时任务
- 如需修改论文查询关键词，可用 `--query` 或设置 `ARXIV_QUERY`

## 🔄 从 ArXiv API 迁移到 OpenAlex

本项目已从 ArXiv 官方 API 迁移到 OpenAlex API，主要原因：

- **更稳定**：OpenAlex 对 GitHub Actions 的 IP 限制更宽松
- **更快速**：通过 Polite Pool（提供邮箱）可获得更高优先级
- **更易用**：返回 JSON 格式，更易解析
- **数据同步**：OpenAlex 实时同步 ArXiv 数据，延迟通常在几小时内

如果你之前使用的是 ArXiv API，只需：
1. 更新代码到最新版本
2. 添加 `OPENALEX_EMAIL` 环境变量
3. 移除 `arxiv` Python 包依赖（已在 requirements.txt 中移除）
