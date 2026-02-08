import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import arxiv
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

try:
    from jinja2 import Environment, StrictUndefined
except ImportError:  # pragma: no cover
    Environment = None
    StrictUndefined = None

PWC_BASE_URL = "https://arxiv.paperswithcode.com/api/v0/papers/"

# ============ 配置常量（不隐私的配置直接写死） ============

# ArXiv 查询关键词
DEFAULT_ARXIV_QUERY = 'abs:"LLM safety" OR abs:"agent safety" OR abs:"AI agent" OR abs:"language model safety" OR abs:"autonomous agent"'

# 每次获取论文数量（会获取更多论文，然后按评分筛选）
DEFAULT_MAX_RESULTS = 50  # 获取 50 篇，筛选出评分 >= 3 的前 20 篇

# 时间范围（小时）：0 表示不限制
DEFAULT_SINCE_HOURS = 0.0

# 最低评分阈值（低于此分数的论文不推送）
MIN_SCORE_THRESHOLD = 3.0

# 最终推送论文数量
FINAL_PUSH_COUNT = 20

# Prompt 模板文件
DEFAULT_PROMPT_FILE = "prompts/deepseek_summary_prompt.zh.j2"

# 模型配置
DEFAULT_MODEL = "glm-4.7-flash"
DEFAULT_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 默认 max_tokens（GLM-4.7 需要更多 tokens 用于推理）
DEFAULT_MAX_TOKENS = 2000


def _strtobool(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _getenv_str(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    return raw


def _getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _resolve_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, path)


def _read_text_file(path: str) -> str:
    resolved = _resolve_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        return f.read()


def _compile_prompt_template(template_text: str):
    if Environment is None or StrictUndefined is None:  # pragma: no cover
        raise RuntimeError("缺少依赖 Jinja2：请先安装 `pip install Jinja2`")
    env = Environment(undefined=StrictUndefined, autoescape=False)
    return env.from_string(template_text)


def get_code_link(arxiv_url: str, session: requests.Session, timeout_s: int = 10) -> Optional[str]:
    """从 PapersWithCode 获取代码链接（若有 official repo）。"""
    arxiv_id = arxiv_url.rstrip("/").split("/")[-1].split("v")[0]
    try:
        resp = session.get(f"{PWC_BASE_URL}{arxiv_id}", timeout=timeout_s)
        if resp.status_code != 200:
            return None
        data = resp.json()
        official = data.get("official")
        if isinstance(official, dict):
            url = official.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url
    except requests.RequestException:
        return None
    except ValueError:
        return None
    return None


def _extract_score(analysis: str) -> float:
    """从分析文本中提取相关性评分（1-5分），默认返回 3.0"""
    match = re.search(r'评分[：:]\s*(\d+(?:\.\d+)?)\s*/\s*5', analysis)
    if match:
        try:
            score = float(match.group(1))
            return max(1.0, min(5.0, score))  # 限制在 1-5 范围内
        except ValueError:
            pass
    return 3.0  # 默认中等评分


def summarize_with_deepseek(
    paper: dict[str, str],
    *,
    prompt_template,
    api_key: str,
    api_url: str,
    model: str,
    max_tokens: int,
    session: requests.Session,
    timeout_s: int = 60,
) -> str:
    """使用 DeepSeek（OpenAI Chat Completions 兼容）进行论文深度总结。"""
    prompt_text = prompt_template.render(**paper).strip()

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严格的学术论文筛选助手，专注于 LLM Safety、Agent Safety 和 AI Agent 领域。请客观评估论文的相关性和创新性。",
            },
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.2,
        "stream": False,
        "max_tokens": max_tokens,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = session.post(api_url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    res_json = resp.json()

    if isinstance(res_json, dict) and "error" in res_json:
        err = res_json.get("error") or {}
        message = err.get("message") if isinstance(err, dict) else None
        raise RuntimeError(f"DeepSeek API 报错: {message or json.dumps(res_json, ensure_ascii=False)}")

    choices = res_json.get("choices") if isinstance(res_json, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"API 未预期响应: {json.dumps(res_json, ensure_ascii=False)}")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None

    # GLM-4.7 等推理模型可能将内容放在 reasoning_content 中
    content = None
    if isinstance(message, dict):
        content = message.get("content")
        # 如果 content 为空，尝试从 reasoning_content 中提取
        if not content or not content.strip():
            reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                print("警告：模型返回的 content 为空，使用 reasoning_content")
                content = reasoning_content

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"API 未返回 content: {json.dumps(res_json, ensure_ascii=False)}")
    return content.strip()


def _feishu_card_payload(title: str, papers: list[dict], footer_note: str) -> dict[str, Any]:
    """生成飞书富文本卡片 payload（支持多篇论文）"""
    elements = []

    for i, paper in enumerate(papers):
        # 提取评分
        analysis = paper['analysis']
        score_match = re.search(r'【相关性】\s*(\d+(?:\.\d+)?)\s*/\s*5', analysis)
        score_text = f"<font color='red'>({score_match.group(1)}/5)</font>" if score_match else ""

        # 标题（带评分）
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**{i+1}/{len(papers)}. <font color='blue'>{paper['title']}</font>** {score_text}"
            }
        })

        # 链接按钮
        actions = [{
            "tag": "button",
            "text": {"tag": "plain_text", "content": "查看论文"},
            "type": "primary",
            "url": paper['url']
        }]
        if paper.get('code_url'):
            actions.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": "查看代码"},
                "type": "default",
                "url": paper['code_url']
            })
        elements.append({"tag": "action", "actions": actions})

        # 合并：问题定义 + 方法核心 + 主要发现
        core_content = []

        problem_match = re.search(r'【问题定义】\s*(.*?)(?=【|$)', analysis, re.DOTALL)
        if problem_match:
            core_content.append(f"<font color='violet'>**【问题定义】**</font>\n{problem_match.group(1).strip()}")

        method_match = re.search(r'【方法核心】\s*(.*?)(?=【|$)', analysis, re.DOTALL)
        if method_match:
            core_content.append(f"<font color='blue'>**【方法核心】**</font>\n{method_match.group(1).strip()}")

        finding_match = re.search(r'【主要发现】\s*(.*?)(?=【|$)', analysis, re.DOTALL)
        if finding_match:
            core_content.append(f"<font color='violet'>**【主要发现】**</font>\n{finding_match.group(1).strip()}")

        if core_content:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "\n".join(core_content)
                }
            })

        # 合并：局限性推测 + 潜在关联
        analysis_content = []

        limitation_match = re.search(r'【局限性推测】\s*(.*?)(?=【|$)', analysis, re.DOTALL)
        if limitation_match:
            analysis_content.append(f"<font color='orange'>**【局限性推测】**</font>\n{limitation_match.group(1).strip()}")

        relation_match = re.search(r'【潜在关联】\s*(.*?)(?=【|$)', analysis, re.DOTALL)
        if relation_match:
            analysis_content.append(f"<font color='green'>**【潜在关联】**</font>\n{relation_match.group(1).strip()}")

        if analysis_content:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "\n".join(analysis_content)
                }
            })

        # 提取一句话结论
        conclusion_match = re.search(r'【一句话结论】\s*(.*?)(?=【|$)', analysis, re.DOTALL)
        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"<font color='blue'>**【一句话结论】**</font>\n{conclusion_text}"
                }
            })

        # 如果不是最后一篇，添加分隔线
        if i < len(papers) - 1:
            elements.append({"tag": "hr"})

    # 添加页脚
    elements.append({"tag": "hr"})
    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": footer_note}]
    })

    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue"
            },
            "elements": elements
        }
    }


def push_to_feishu(
    papers: list[dict],
    *,
    webhook: str,
    session: requests.Session,
    title: str,
    footer_note: str,
    timeout_s: int = 15,
) -> None:
    """发送飞书富文本卡片（失败会抛异常）。

    Args:
        papers: 论文列表，每个元素包含 title, url, code_url, analysis
        webhook: 飞书 Webhook 地址
        session: requests.Session 对象
        title: 卡片标题
        footer_note: 页脚文本
        timeout_s: 超时时间（秒）
    """
    headers = {"Content-Type": "application/json"}
    payload = _feishu_card_payload(title=title, papers=papers, footer_note=footer_note)
    resp = session.post(webhook, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        code = data.get("code", data.get("StatusCode"))
        if code not in (0, "0", None):
            raise RuntimeError(f"飞书返回错误: {json.dumps(data, ensure_ascii=False)}")


def _write_github_step_summary(markdown: str) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown.rstrip() + "\n")
    except OSError:
        return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest arXiv papers, summarize, and push to Feishu.")
    parser.add_argument("--query", default=_getenv_str("ARXIV_QUERY", DEFAULT_ARXIV_QUERY))
    parser.add_argument("--max-results", type=int, default=_getenv_int("MAX_RESULTS", DEFAULT_MAX_RESULTS))
    parser.add_argument("--since-hours", type=float, default=_getenv_float("SINCE_HOURS", DEFAULT_SINCE_HOURS))

    parser.add_argument("--feishu-webhook", default=_getenv_str("FEISHU_WEBHOOK"))
    parser.add_argument("--per-paper", action="store_true", default=_strtobool(os.getenv("FEISHU_PER_PAPER")))

    parser.add_argument("--deepseek-api-key", default=_getenv_str("DEEPSEEK_API_KEY"))
    parser.add_argument("--deepseek-model", default=_getenv_str("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--deepseek-api-url", default=_getenv_str("DEEPSEEK_API_URL"))  # 如果未指定，将根据模型自动选择
    parser.add_argument("--deepseek-max-tokens", type=int, default=_getenv_int("DEEPSEEK_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    parser.add_argument("--skip-llm", action="store_true", default=_strtobool(os.getenv("SKIP_LLM")))
    parser.add_argument("--prompt-file", default=_getenv_str("PROMPT_FILE", DEFAULT_PROMPT_FILE))

    parser.add_argument("--dry-run", action="store_true", default=_strtobool(os.getenv("DRY_RUN")))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    session = requests.Session()

    # 如果未指定 API URL，使用默认值
    if not args.deepseek_api_url:
        args.deepseek_api_url = DEFAULT_API_URL
        print(f"使用默认 API 端点：{args.deepseek_api_url}（模型：{args.deepseek_model}）")

    if not args.dry_run and not args.feishu_webhook:
        print("缺少 FEISHU_WEBHOOK：请在环境变量或参数中设置 --feishu-webhook。", file=sys.stderr)
        return 2
    if not args.skip_llm and not args.deepseek_api_key:
        print("缺少 DEEPSEEK_API_KEY：请在环境变量或参数中设置 --deepseek-api-key，或使用 --skip-llm。", file=sys.stderr)
        return 2

    prompt_template = None
    if not args.skip_llm:
        try:
            template_text = _read_text_file(args.prompt_file)
            prompt_template = _compile_prompt_template(template_text)
        except Exception as e:
            print(f"无法加载 Prompt 模板（{args.prompt_file}）：{e}", file=sys.stderr)
            return 2

    print("正在搜集最新论文...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=args.query,
        max_results=args.max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = list(client.results(search))
    if args.since_hours > 0:
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(hours=float(args.since_hours))
        results = [r for r in results if getattr(r, "published", None) and r.published.replace(tzinfo=timezone.utc) >= threshold]

    if not results:
        msg = "今日暂无新论文。"
        print(msg)
        _write_github_step_summary(f"## ArXiv 每日推送\n\n{msg}\n")
        if not args.dry_run and args.feishu_webhook:
            push_to_feishu(
                msg,
                webhook=args.feishu_webhook,
                session=session,
                title=f"🚀 ArXiv {datetime.now().strftime('%m-%d')}",
                footer_note="自动生成：无新论文",
            )
        return 0

    # 第一步：分析所有论文并提取评分
    paper_data: list[dict] = []
    total = len(results)
    for i, res in enumerate(results, start=1):
        print(f"正在分析第 {i}/{total} 篇: {res.title}")
        code_url = get_code_link(res.entry_id, session=session)
        paper_info = {
            "title": res.title.strip(),
            "summary": (res.summary or "").replace("\n", " ").strip(),
            "url": res.entry_id,
        }

        if args.skip_llm:
            analysis = f"【摘要（未调用 LLM）】\n{paper_info['summary']}\n"
            score = 3.0
        else:
            try:
                analysis = summarize_with_deepseek(
                    paper_info,
                    prompt_template=prompt_template,
                    api_key=args.deepseek_api_key,
                    api_url=args.deepseek_api_url,
                    model=args.deepseek_model,
                    max_tokens=args.deepseek_max_tokens,
                    session=session,
                )
                # 打印 LLM 返回的原始内容（用于调试）
                print("\n=== LLM 返回内容 ===")
                print(analysis)
                print("===================\n")
                score = _extract_score(analysis)
            except Exception as e:
                print(f"LLM 调用失败: {str(e)}")
                analysis = f"【LLM 解析失败】{str(e)}\n\n【摘要】{paper_info['summary']}"
                score = 3.0

        paper_data.append({
            "title": paper_info["title"],
            "url": paper_info["url"],
            "code_url": code_url,
            "analysis": analysis,
            "score": score,
        })

    # 第二步：按评分从高到低排序
    paper_data.sort(key=lambda x: x["score"], reverse=True)

    # 第三步：过滤低分论文（评分 < MIN_SCORE_THRESHOLD 的不推送）
    filtered_papers = [p for p in paper_data if p["score"] >= MIN_SCORE_THRESHOLD]

    if not filtered_papers:
        msg = f"今日无高相关性论文（所有论文评分 < {MIN_SCORE_THRESHOLD}）。"
        print(msg)
        _write_github_step_summary(f"## ArXiv 每日推送\n\n{msg}\n")
        # 不推送空消息到飞书
        return 0

    # 第四步：只保留前 FINAL_PUSH_COUNT 篇论文
    final_papers = filtered_papers[:FINAL_PUSH_COUNT]
    print(f"筛选后共 {len(filtered_papers)} 篇高分论文，推送前 {len(final_papers)} 篇")

    # 第五步：生成推送内容
    date_label = datetime.now().strftime("%m-%d")
    card_title = f"🚀 ArXiv {date_label}"
    footer_note = f"自动生成 | 共 {len(final_papers)} 篇高相关性论文"

    # 生成 GitHub Step Summary
    summary_blocks = []
    for i, paper in enumerate(final_papers, start=1):
        code_md = f" | [💻 代码]({paper['code_url']})" if paper.get('code_url') else ""
        header = f"### {i}/{len(final_papers)}. {paper['title']}\n🔗 [原文]({paper['url']}){code_md}\n"
        summary_blocks.append(header + paper['analysis'].strip() + "\n")

    summary_md = f"## ArXiv 每日推送 ({date_label})\n\n" + "\n---\n\n".join(summary_blocks)
    _write_github_step_summary(summary_md)

    if args.dry_run:
        print(summary_md)
        return 0

    # 推送到飞书
    if not args.feishu_webhook:
        print("未配置飞书 Webhook，跳过推送")
        return 0

    if args.per_paper:
        # 每篇论文单独推送
        for i, paper in enumerate(final_papers, start=1):
            push_to_feishu(
                [paper],
                webhook=args.feishu_webhook,
                session=session,
                title=f"🚀 ArXiv {date_label} ({i}/{len(final_papers)})",
                footer_note=footer_note,
            )
    else:
        # 合并推送
        push_to_feishu(
            final_papers,
            webhook=args.feishu_webhook,
            session=session,
            title=card_title,
            footer_note=footer_note,
        )

    print("推送成功！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
