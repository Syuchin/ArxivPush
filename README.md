# 📚 ArXiv 每日论文推送助手

自动抓取 ArXiv 最新 AI 论文，使用 DeepSeek 进行深度分析，并推送到飞书。

## ✨ 功能特性

- 🔍 **自动抓取**：每日自动获取 ArXiv 最新 LLM / AI Agent / Deep Learning 相关论文
- 🤖 **AI 深度分析**：调用 DeepSeek API 生成结构化中文解读：
  - 【快速抓要点】核心问题与方法
  - 【逻辑推导】起承转合还原作者思路
  - 【技术细节】关键实现细节
  - 【局限性】潜在不足
  - 【专业知识解释】术语科普
- 💻 **代码链接**：自动从 PapersWithCode 匹配开源代码
- 📱 **飞书推送**：生成精美富文本卡片推送至飞书群

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置（推荐用环境变量）

通过环境变量提供飞书 Webhook 与 DeepSeek Key（避免把密钥写进代码/仓库）：

```bash
export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/你的Webhook地址"
export DEEPSEEK_API_KEY="你的DeepSeek API Key"
# 可选：不填则默认 https://api.deepseek.com/v1/chat/completions
export DEEPSEEK_API_URL="https://api.deepseek.com/v1/chat/completions"
```

你也可以把上述变量写到项目根目录的 `.env`（参考 `.env.example`），脚本会自动加载。

默认 Prompt 模板在 `prompts/deepseek_summary_prompt.zh.j2`（Jinja2），可用 `{{ title }}`、`{{ summary }}`（以及 `url`）等变量自由修改。

- 飞书 Webhook：在飞书群设置 → 添加机器人 → 自定义机器人 → 获取 Webhook 地址
- DeepSeek API Key：在 DeepSeek 开放平台 获取

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

1. 把本项目推到你的 GitHub 仓库
2. 在仓库中配置 Secrets：
   - `FEISHU_WEBHOOK`
   - `DEEPSEEK_API_KEY`
3. （可选）配置 Variables：
   - `ARXIV_QUERY`、`MAX_RESULTS`、`SINCE_HOURS`
   - `DEEPSEEK_API_URL`、`DEEPSEEK_MODEL`、`DEEPSEEK_MAX_TOKENS`
   - `FEISHU_PER_PAPER`
   - `TZ`（例如 `Asia/Shanghai`，用于卡片标题日期显示）
4. 在 Actions 页面启用工作流；也可用 `workflow_dispatch` 手动跑一次验证

> 注意：GitHub Actions 的 cron 使用 UTC 时间；如需改运行时间，编辑 `.github/workflows/arxiv-daily.yml` 里的 `cron`。

### 5. 注意事项

- 确保网络可访问 ArXiv、DeepSeek API 和飞书服务器
- 建议先手动运行测试，确认配置无误后再设置定时任务
- 如需修改论文查询关键词，可用 `--query` 或设置 `ARXIV_QUERY`
