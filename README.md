# 📚 ArXiv 每日论文推送助手

自动抓取 ArXiv 最新 AI 论文，使用 LLM 进行深度分析，并推送到飞书。

## ✨ 功能特性

- 🔍 **自动抓取**：每日自动获取 ArXiv 最新 LLM Safety / Agent Safety / AI Agent 相关论文
- 🤖 **AI 深度分析**：调用 LLM API（支持智谱 GLM、DeepSeek 等）生成结构化中文解读：
  - 【相关性评分】：1-5 分评估论文与研究领域的相关性
  - 【动机】：研究背景和要解决的问题
  - 【方法】：核心技术方案
  - 【发现】：主要结果和贡献
  - 【创新点评价】：idea 的新颖性和实用价值
- 📊 **智能排序**：按相关性评分自动排序，优先推送高分论文
- 🎯 **智能过滤**：自动过滤低相关性论文（评分 < 3 分）
- 💻 **代码链接**：自动从 PapersWithCode 匹配开源代码
- 📱 **飞书推送**：生成精美富文本卡片推送至飞书群

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置（推荐用环境变量）

通过环境变量提供飞书 Webhook 与 API Key（避免把密钥写进代码/仓库）：

```bash
export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/你的Webhook地址"
export DEEPSEEK_API_KEY="你的API Key"
# 智谱 GLM API（推荐）
export DEEPSEEK_API_URL="https://open.bigmodel.cn/api/paas/v4/chat/completions"
# 或使用 DeepSeek API
# export DEEPSEEK_API_URL="https://api.deepseek.com/v1/chat/completions"
```

你也可以把上述变量写到项目根目录的 `.env`（参考 `.env.example`），脚本会自动加载。

默认 Prompt 模板在 `prompts/deepseek_summary_prompt.zh.j2`（Jinja2），可用 `{{ title }}`、`{{ summary }}`（以及 `url`）等变量自由修改。

- 飞书 Webhook：在飞书群设置 → 添加机器人 → 自定义机器人 → 获取 Webhook 地址
- 智谱 GLM API Key：在 [智谱开放平台](https://open.bigmodel.cn/) 获取
- DeepSeek API Key：在 [DeepSeek 开放平台](https://platform.deepseek.com/) 获取

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

   添加以下 Secrets：
   - `FEISHU_WEBHOOK`：你的飞书 Webhook 地址
   - `DEEPSEEK_API_KEY`：你的 API Key（智谱 GLM 或 DeepSeek）
   - `DEEPSEEK_API_URL`：API 地址
     - 智谱 GLM：`https://open.bigmodel.cn/api/paas/v4/chat/completions`
     - DeepSeek：`https://api.deepseek.com/v1/chat/completions`
   - `DEEPSEEK_MODEL`：模型名称（如 `glm-4-flash` 或 `deepseek-chat`）
   - `ARXIV_QUERY`：查询关键词（如 `abs:"LLM safety" OR abs:"agent safety" OR abs:"AI agent"`）
   - `MAX_RESULTS`：每次获取论文数量（如 `10`）

3. **启用 GitHub Actions**

   进入仓库页面 → Actions → 启用工作流

   可以点击 "Run workflow" 手动触发一次测试

4. **定时运行**

   工作流默认每天 UTC 00:00（北京时间 08:00）自动运行

   如需修改时间，编辑 `.github/workflows/arxiv-daily.yml` 中的 `cron` 表达式

> 注意：GitHub Actions 的 cron 使用 UTC 时间；如需改运行时间，编辑 `.github/workflows/arxiv-daily.yml` 里的 `cron`。

### 5. 注意事项

- 确保网络可访问 ArXiv、DeepSeek API 和飞书服务器
- 建议先手动运行测试，确认配置无误后再设置定时任务
- 如需修改论文查询关键词，可用 `--query` 或设置 `ARXIV_QUERY`
