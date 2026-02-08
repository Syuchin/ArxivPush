import argparse
import json
import os
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
DEFAULT_DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_ARXIV_QUERY = 'abs:LLM OR abs:"AI Agent" OR abs:"Deep Learning"'
DEFAULT_PROMPT_FILE = "prompts/deepseek_summary_prompt.zh.j2"


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
                "content": "你擅长把 AI 论文总结成结构化要点，保持严谨，不胡编。",
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
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"API 未返回 content: {json.dumps(res_json, ensure_ascii=False)}")
    return content.strip()


def _feishu_card_payload(title: str, markdown: str, footer_note: str) -> dict[str, Any]:
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "orange",
            },
            "elements": [
                {"tag": "markdown", "content": markdown},
                {"tag": "hr"},
                {"tag": "note", "elements": [{"tag": "plain_text", "content": footer_note}]},
            ],
        },
    }


def push_to_feishu(
    markdown: str,
    *,
    webhook: str,
    session: requests.Session,
    title: str,
    footer_note: str,
    timeout_s: int = 15,
) -> None:
    """发送飞书富文本卡片（失败会抛异常）。"""
    headers = {"Content-Type": "application/json"}
    payload = _feishu_card_payload(title=title, markdown=markdown, footer_note=footer_note)
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
    parser.add_argument("--max-results", type=int, default=_getenv_int("MAX_RESULTS", 3))
    parser.add_argument("--since-hours", type=float, default=_getenv_float("SINCE_HOURS", 0.0))

    parser.add_argument("--feishu-webhook", default=_getenv_str("FEISHU_WEBHOOK"))
    parser.add_argument("--per-paper", action="store_true", default=_strtobool(os.getenv("FEISHU_PER_PAPER")))

    parser.add_argument("--deepseek-api-key", default=_getenv_str("DEEPSEEK_API_KEY"))
    parser.add_argument("--deepseek-api-url", default=_getenv_str("DEEPSEEK_API_URL", DEFAULT_DEEPSEEK_API_URL))
    parser.add_argument("--deepseek-model", default=_getenv_str("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--deepseek-max-tokens", type=int, default=_getenv_int("DEEPSEEK_MAX_TOKENS", 900))
    parser.add_argument("--skip-llm", action="store_true", default=_strtobool(os.getenv("SKIP_LLM")))
    parser.add_argument("--prompt-file", default=_getenv_str("PROMPT_FILE", DEFAULT_PROMPT_FILE))

    parser.add_argument("--dry-run", action="store_true", default=_strtobool(os.getenv("DRY_RUN")))
    return parser.parse_args()


def _format_paper_block(idx: int, total: int, *, title: str, url: str, code_url: Optional[str], analysis: str) -> str:
    code_md = f" | [💻 代码]({code_url})" if code_url else ""
    header = f"### {idx}/{total}. {title}\n🔗 [原文]({url}){code_md}\n"
    return header + analysis.strip() + "\n"


def main() -> int:
    args = _parse_args()
    session = requests.Session()

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

    blocks: list[str] = []
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
            except Exception as e:
                analysis = f"【LLM 解析失败】{str(e)}\n\n【摘要】{paper_info['summary']}"

        blocks.append(
            _format_paper_block(
                i,
                total,
                title=paper_info["title"],
                url=paper_info["url"],
                code_url=code_url,
                analysis=analysis,
            )
        )

    date_label = datetime.now().strftime("%m-%d")
    footer = "基于 DeepSeek 自动生成（仅供学习参考）"

    if args.dry_run:
        combined = "\n\n---\n\n".join(blocks).strip() + "\n"
        print(combined)
        _write_github_step_summary(f"## ArXiv {date_label}\n\n" + combined)
        return 0

    if args.per_paper:
        for block in blocks:
            push_to_feishu(
                block,
                webhook=args.feishu_webhook,
                session=session,
                title=f"🚀 ArXiv {date_label}",
                footer_note=footer,
            )
    else:
        combined = "\n\n---\n\n".join(blocks).strip()
        push_to_feishu(
            combined,
            webhook=args.feishu_webhook,
            session=session,
            title=f"🚀 ArXiv {date_label}",
            footer_note=footer,
        )

    _write_github_step_summary(f"## ArXiv {date_label}\n\n" + "\n\n---\n\n".join(blocks).strip() + "\n")
    print("推送成功！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
