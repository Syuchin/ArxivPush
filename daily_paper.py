import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from threading import Semaphore
from typing import Any, Optional

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
OPENALEX_BASE_URL = "https://api.openalex.org/works"
PUSHED_IDS_FILE = "pushed_ids.txt"  # 存储已推送论文 ID 的文件

# ============ 配置常量（不隐私的配置直接写死） ============

# ArXiv 查询关键词（用于 OpenAlex API）
# 扩大范围：包含 LLM、Agent 和安全相关的论文
DEFAULT_ARXIV_QUERY = 'LLM OR agent OR language model OR safety OR security OR robustness'

# 每次获取论文数量（会获取更多论文，然后按评分筛选）
DEFAULT_MAX_RESULTS = 50  # 获取 50 篇，筛选出高分论文

# 时间范围（小时）：获取最近 N 小时内发布的论文
# 设置为 96 小时（4 天）以覆盖周末（ArXiv 周末不更新）
# 通过去重机制，每天只推送新论文，实现真正的"每日更新"
DEFAULT_SINCE_HOURS = 96.0

# 最低评分阈值（低于此分数的论文不推送）
MIN_SCORE_THRESHOLD = 3.0

# 5 分论文全部推送，其他高分论文限制数量
PERFECT_SCORE = 5.0  # 满分论文
MAX_NON_PERFECT_PAPERS = 20  # 非满分论文最多推送数量

# Prompt 模板文件
DEFAULT_PROMPT_FILE = "prompts/deepseek_summary_prompt.zh.j2"

# 模型配置
DEFAULT_MODEL = "glm-4.7"
DEFAULT_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 默认 max_tokens（GLM-4.7 需要更多 tokens 用于推理）
DEFAULT_MAX_TOKENS = 2000

# 并发处理线程数（严格控制同时进行的 API 请求数）
MAX_WORKERS = 3

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒

# 请求间隔（秒）- 避免瞬间发送过多请求
REQUEST_INTERVAL = 0.5

# OpenAlex API 配置
OPENALEX_PER_PAGE = 50  # 每次请求的结果数量
OPENALEX_TIMEOUT = 30  # 请求超时时间（秒）

# 预编译正则：匹配【相关性】X/5 格式的评分
_SCORE_RE = re.compile(r'【相关性】\s*(\d+(?:\.\d+)?)\s*/\s*5')

# 预编译正则：匹配【标签名】内容 格式的分析段落
_SECTION_RE = re.compile(r'【([^】]+)】\s*(.*?)(?=【|$)', re.DOTALL)


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
    # 匹配【相关性】X/5 格式
    match = re.search(r'【相关性】\s*(\d+(?:\.\d+)?)\s*/\s*5', analysis)
    if match:
        try:
            score = float(match.group(1))
            return max(1.0, min(5.0, score))  # 限制在 1-5 范围内
        except ValueError:
            pass
    return 3.0  # 默认中等评分


def _process_single_paper(
    res,
    index: int,
    total: int,
    *,
    session: requests.Session,
    skip_llm: bool,
    prompt_template,
    api_key: str,
    api_url: str,
    model: str,
    max_tokens: int,
    semaphore: Semaphore,
) -> dict:
    """处理单篇论文（用于并发）"""
    print(f"正在分析第 {index}/{total} 篇: {res.title}")

    code_url = get_code_link(res.entry_id, session=session)
    paper_info = {
        "title": res.title.strip(),
        "summary": (res.summary or "").replace("\n", " ").strip(),
        "url": res.entry_id,
    }

    if skip_llm:
        analysis = f"【摘要（未调用 LLM）】\n{paper_info['summary']}\n"
        score = 3.0
    else:
        try:
            # 使用信号量控制并发 API 请求数
            with semaphore:
                # 添加请求间隔，避免瞬间发送过多请求
                time.sleep(REQUEST_INTERVAL)
                analysis = summarize_with_deepseek(
                    paper_info,
                    prompt_template=prompt_template,
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    max_tokens=max_tokens,
                    session=session,
                )
            # 打印 LLM 返回的原始内容（用于调试）
            print(f"\n=== 论文 {index} LLM 返回内容 ===")
            print(analysis)
            print("===================\n")
            score = _extract_score(analysis)
        except Exception as e:
            print(f"论文 {index} LLM 调用失败: {str(e)}")
            analysis = f"【LLM 解析失败】{str(e)}\n\n【摘要】{paper_info['summary']}"
            score = 3.0

    return {
        "title": paper_info["title"],
        "url": paper_info["url"],
        "code_url": code_url,
        "analysis": analysis,
        "score": score,
    }


def summarize_with_deepseek(
    paper: dict[str, str],
    *,
    prompt_template,
    api_key: str,
    api_url: str,
    model: str,
    max_tokens: int,
    session: requests.Session,
    timeout_s: int = 120,  # 增加超时时间到 120 秒
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
        "temperature": 1.0,
        "stream": False,
        "thinking": {
            "type": "disabled"
        },
        "max_tokens": max_tokens,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # 重试逻辑
    for attempt in range(MAX_RETRIES):
        try:
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

        except (requests.exceptions.Timeout, requests.exceptions.HTTPError, Exception) as e:
            is_last_attempt = attempt == MAX_RETRIES - 1

            # 429 错误使用指数退避，其他错误使用线性退避
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                wait_time = RETRY_DELAY * (attempt + 2) * 2
                error_msg = "API 限流"
            else:
                wait_time = RETRY_DELAY * (attempt + 1)
                error_msg = "请求超时" if isinstance(e, requests.exceptions.Timeout) else f"请求失败: {str(e)}"

            if not is_last_attempt:
                print(f"{error_msg}，{wait_time} 秒后重试（第 {attempt + 1}/{MAX_RETRIES} 次）...")
                time.sleep(wait_time)
            else:
                # 最后一次尝试失败，抛出异常
                raise


def _extract_section(analysis: str, section_name: str) -> Optional[str]:
    """从分析文本中提取指定段落的内容"""
    match = re.search(rf'【{re.escape(section_name)}】\s*(.*?)(?=【|$)', analysis, re.DOTALL)
    return match.group(1).strip() if match else None


def _feishu_card_payload(title: str, papers: list[dict], footer_note: str) -> dict[str, Any]:
    """生成飞书富文本卡片 payload（支持多篇论文）"""
    elements = []

    for i, paper in enumerate(papers):
        analysis = paper['analysis']

        # 提取评分
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
        core_sections = [
            ("问题定义", "violet"),
            ("方法核心", "blue"),
            ("主要发现", "violet"),
        ]
        core_content = []
        for section_name, color in core_sections:
            content = _extract_section(analysis, section_name)
            if content:
                core_content.append(f"<font color='{color}'>**【{section_name}】**</font>\n{content}")

        if core_content:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "\n".join(core_content)
                }
            })

        # 合并：局限性推测 + 潜在关联
        analysis_sections = [
            ("局限性推测", "orange"),
            ("潜在关联", "green"),
        ]
        analysis_content = []
        for section_name, color in analysis_sections:
            content = _extract_section(analysis, section_name)
            if content:
                analysis_content.append(f"<font color='{color}'>**【{section_name}】**</font>\n{content}")

        if analysis_content:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "\n".join(analysis_content)
                }
            })

        # 提取一句话结论
        conclusion = _extract_section(analysis, "一句话结论")
        if conclusion:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"<font color='blue'>**【一句话结论】**</font>\n{conclusion}"
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




def fetch_papers_from_openalex(
    query: str,
    max_results: int,
    since_hours: float,
    email: str,
    session: requests.Session,
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    从 OpenAlex API 获取 ArXiv 论文

    Args:
        query: 搜索关键词
        max_results: 最大结果数
        since_hours: 只获取最近 N 小时内的论文（0 表示不限制）
        email: 用于 Polite Pool 的邮箱地址
        session: requests.Session 对象
        api_key: OpenAlex API Key（可选，用于访问高级功能如 from_created_date）

    Returns:
        论文列表，每个论文是一个字典，包含 title, summary, entry_id, published 等字段
    """
    print(f"正在从 OpenAlex API 查询论文（关键词：{query}）...")

    # 构建查询参数
    # 如果有 API Key，使用 from_created_date（更准确）
    # 否则使用 from_publication_date
    use_created_date = api_key is not None
    date_filter_type = 'from_created_date' if use_created_date else 'from_publication_date'
    sort_field = 'created_date' if use_created_date else 'publication_date'

    params = {
        'filter': f'indexed_in:arxiv,title.search:{query}',
        'sort': f'{sort_field}:desc',
        'mailto': email,
        'per_page': min(OPENALEX_PER_PAGE, max_results),
    }

    # 如果指定了时间范围，添加日期过滤
    if since_hours > 0:
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(hours=since_hours)
        # OpenAlex 使用 YYYY-MM-DD 格式
        from_date = threshold.strftime('%Y-%m-%d')
        params['filter'] += f',{date_filter_type}:{from_date}'
        if use_created_date:
            print(f"过滤条件：只获取 {from_date} 之后添加到 OpenAlex 的论文（使用 API Key）")
        else:
            print(f"过滤条件：只获取 {from_date} 之后发布的论文")

    papers = []
    page = 1

    while len(papers) < max_results:
        try:
            params['page'] = page

            # 如果有 API Key，添加到请求头
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            response = session.get(
                OPENALEX_BASE_URL,
                params=params,
                headers=headers,
                timeout=OPENALEX_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            if not results:
                break

            for work in results:
                if len(papers) >= max_results:
                    break

                # 提取 ArXiv ID 和 URL
                arxiv_id = None
                arxiv_url = None

                # 从 ids 字段中提取 ArXiv ID
                ids = work.get('ids', {})
                if 'arxiv' in ids:
                    arxiv_url = ids['arxiv']
                    # 从 URL 中提取 ID
                    if arxiv_url:
                        match = re.search(r'arxiv\.org/abs/(\S+)', arxiv_url)
                        if match:
                            arxiv_id = match.group(1)

                # 如果没有找到 ArXiv URL，尝试从 locations 中查找
                if not arxiv_url:
                    locations = work.get('locations', [])
                    for loc in locations:
                        landing_page = loc.get('landing_page_url', '')
                        if 'arxiv.org' in landing_page:
                            arxiv_url = landing_page
                            match = re.search(r'arxiv\.org/abs/(\S+)', landing_page)
                            if match:
                                arxiv_id = match.group(1)
                            break

                # 如果还是没有找到，跳过这篇论文
                if not arxiv_url:
                    continue

                # 提取摘要
                abstract = work.get('abstract', '')
                if not abstract:
                    # 如果没有摘要，尝试使用 abstract_inverted_index
                    inv_index = work.get('abstract_inverted_index', {})
                    if inv_index:
                        # 重建摘要文本
                        words = []
                        for word, positions in inv_index.items():
                            for pos in positions:
                                words.append((pos, word))
                        words.sort()
                        abstract = ' '.join([w[1] for w in words])

                # 提取发布日期
                pub_date_str = work.get('publication_date')
                published = None
                if pub_date_str:
                    try:
                        published = datetime.strptime(pub_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass

                # 提取 OpenAlex ID（用于去重）
                openalex_id = work.get('id', '')

                # 构建与 ArXiv API 兼容的论文对象（使用简单的字典模拟 arxiv.Result）
                class PaperResult:
                    def __init__(self, title, summary, entry_id, published, openalex_id):
                        self.title = title
                        self.summary = summary
                        self.entry_id = entry_id
                        self.published = published
                        self.openalex_id = openalex_id  # 用于去重

                paper = PaperResult(
                    title=work.get('display_name', '').strip(),
                    summary=abstract.strip() if abstract else '',
                    entry_id=arxiv_url,
                    published=published,
                    openalex_id=openalex_id,
                )

                papers.append(paper)

            # 检查是否还有更多结果
            meta = data.get('meta', {})
            if page * params['per_page'] >= meta.get('count', 0):
                break

            page += 1
            time.sleep(REQUEST_INTERVAL)  # 避免请求过快

        except requests.exceptions.RequestException as e:
            print(f"OpenAlex API 请求失败: {str(e)}", file=sys.stderr)
            break
        except Exception as e:
            print(f"解析 OpenAlex 响应时出错: {str(e)}", file=sys.stderr)
            break

    print(f"从 OpenAlex 获取到 {len(papers)} 篇论文")
    return papers


def _write_github_step_summary(markdown: str) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown.rstrip() + "\n")
    except OSError:
        pass


def load_pushed_ids() -> set[str]:
    """加载已推送的论文 ID 列表"""
    if not os.path.exists(PUSHED_IDS_FILE):
        return set()
    try:
        with open(PUSHED_IDS_FILE, 'r', encoding='utf-8') as f:
            # 只保留最近 500 条记录，避免文件过大
            ids = [line.strip() for line in f if line.strip()]
            return set(ids[-500:])
    except Exception as e:
        print(f"读取已推送 ID 文件失败: {e}", file=sys.stderr)
        return set()


def save_pushed_ids(ids: set[str]) -> None:
    """保存已推送的论文 ID 列表"""
    try:
        # 只保留最近 500 条记录
        ids_list = list(ids)[-500:]
        with open(PUSHED_IDS_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ids_list))
        print(f"已更新推送记录文件，当前记录数: {len(ids_list)}")
    except Exception as e:
        print(f"保存已推送 ID 文件失败: {e}", file=sys.stderr)


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

    # OpenAlex API 配置
    parser.add_argument("--openalex-email", default=_getenv_str("OPENALEX_EMAIL"), help="Email for OpenAlex Polite Pool")
    parser.add_argument("--openalex-api-key", default=_getenv_str("OPENALEX_API_KEY"), help="OpenAlex API Key (optional, for advanced features)")

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
    if not args.openalex_email:
        print("缺少 OPENALEX_EMAIL：请在环境变量或参数中设置 --openalex-email。", file=sys.stderr)
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
    # 使用 OpenAlex API 获取论文
    results = fetch_papers_from_openalex(
        query=args.query,
        max_results=args.max_results,
        since_hours=args.since_hours,
        email=args.openalex_email,
        session=session,
        api_key=args.openalex_api_key,
    )

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

    # 加载已推送的论文 ID 并进行去重
    pushed_ids = load_pushed_ids()
    original_count = len(results)

    # 过滤掉已推送的论文
    results = [r for r in results if getattr(r, 'openalex_id', '') not in pushed_ids]

    if len(results) < original_count:
        print(f"去重：过滤掉 {original_count - len(results)} 篇已推送的论文，剩余 {len(results)} 篇")

    if not results:
        msg = "今日无新论文（所有论文均已推送过）。"
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

    # 第一步：并发分析所有论文并提取评分
    paper_data: list[dict] = []
    total = len(results)

    print(f"开始并发分析 {total} 篇论文（并发数：{MAX_WORKERS}）...")

    # 创建信号量，严格控制同时进行的 API 请求数
    api_semaphore = Semaphore(MAX_WORKERS)

    # ThreadPoolExecutor 的 max_workers 设为 MAX_WORKERS 即可
    # Semaphore 已经控制了并发 API 请求数，不需要额外的线程
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(
                _process_single_paper,
                res,
                i,
                total,
                session=session,
                skip_llm=args.skip_llm,
                prompt_template=prompt_template,
                api_key=args.deepseek_api_key,
                api_url=args.deepseek_api_url,
                model=args.deepseek_model,
                max_tokens=args.deepseek_max_tokens,
                semaphore=api_semaphore,
            ): i
            for i, res in enumerate(results, start=1)
        }

        # 收集结果
        for future in as_completed(future_to_index):
            try:
                paper = future.result()
                paper_data.append(paper)
            except Exception as e:
                index = future_to_index[future]
                print(f"处理论文 {index} 时发生错误: {str(e)}")
                # 继续处理其他论文

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

    # 第四步：分离满分论文和其他高分论文
    # 满分论文（5 分）全部推送，其他高分论文最多推送 MAX_NON_PERFECT_PAPERS 篇
    perfect_papers = [p for p in filtered_papers if p["score"] >= PERFECT_SCORE]
    other_papers = [p for p in filtered_papers if p["score"] < PERFECT_SCORE]

    # 组合：所有满分论文 + 最多 MAX_NON_PERFECT_PAPERS 篇其他高分论文
    final_papers = perfect_papers + other_papers[:MAX_NON_PERFECT_PAPERS]

    print(f"筛选后共 {len(filtered_papers)} 篇高分论文：")
    print(f"  - 满分论文（5 分）：{len(perfect_papers)} 篇（全部推送）")
    print(f"  - 其他高分论文（3-4.9 分）：{len(other_papers)} 篇（推送前 {min(len(other_papers), MAX_NON_PERFECT_PAPERS)} 篇）")
    print(f"  - 最终推送：{len(final_papers)} 篇")

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

    # 保存已推送的论文 ID
    # 从 results 中提取所有被处理的论文的 OpenAlex ID
    processed_ids = {getattr(r, 'openalex_id', '') for r in results if getattr(r, 'openalex_id', '')}
    if processed_ids:
        pushed_ids.update(processed_ids)
        save_pushed_ids(pushed_ids)

    print("推送成功！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
