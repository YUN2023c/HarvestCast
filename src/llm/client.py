# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""LLM API 调用层（仅标准库 urllib，无 SDK 依赖）。

支持两种主流接口形态，通过 settings.llm.api_style 选择：

- ``chat``（默认）：POST {base_url}/chat/completions，使用 response_format.json_schema
  强制结构化输出。OpenAI、DeepSeek、通义、智谱、Moonshot、vLLM 等兼容端点大多可用。
- ``responses``：POST {base_url}/responses，可叠加官方 web_search 工具，实现「联网 + 结构化」
  一步到位（OpenAI Responses API 形态）。

⚠️ 能力边界（务必写清，避免误导）：
纯 chat/completions 接口本身不联网。所谓「自动模式」= 模型自带检索能力（如带搜索的
托管模型 / responses + web_search 工具），或由上游网关代检索。若所用模型不支持检索，
请改用人工模式（把提示词粘贴到任意联网模型），两条路径的校验与产出结构完全一致。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from ..core.config import api_key
from . import prompt_builder as pb
from .jsonschema_lite import validate


class LLMError(RuntimeError):
    pass


def _post(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")[:600]
        except Exception:  # noqa: BLE001
            pass
        raise LLMError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"网络不可达: {exc.reason}") from exc


def _extract_json_text(content: str) -> str:
    """容错：模型偶尔仍会包一层 ```json 围栏或前后说明。"""
    s = content.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        return s[start:end + 1]
    return s


def call_chat(rendered: Dict[str, Any], settings: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    llm = settings.get("llm", {})
    key = api_key(settings)
    if not key:
        return None, {"status": "no_key",
                      "message": "未配置 API Key，请设置环境变量 HARVESTCAST_API_KEY 或改用人工模式"}
    base = llm.get("base_url", "").rstrip("/")
    style = llm.get("api_style", "chat")
    schema = rendered["schema"]
    timeout = int(llm.get("timeout_seconds", 120))
    retries = int(llm.get("retry", 2))
    last_err = ""

    for attempt in range(retries + 1):
        try:
            if style == "responses":
                payload: Dict[str, Any] = {
                    "model": llm.get("model"),
                    "input": [
                        {"role": "system", "content": rendered["system"]},
                        {"role": "user", "content": rendered["user"]},
                    ],
                    "text": {"format": {"type": "json_schema", "name": "fundamental_extraction",
                                        "strict": True, "schema": schema}},
                }
                if llm.get("enable_web_search"):
                    payload["tools"] = [{"type": llm.get("search_tool", "web_search_preview")}]
                resp = _post(f"{base}/responses", payload,
                             {"Authorization": f"Bearer {key}"}, timeout)
                text = _responses_text(resp)
            else:
                payload = {
                    "model": llm.get("model"),
                    "messages": [
                        {"role": "system", "content": rendered["system"]},
                        {"role": "user", "content": rendered["user"]},
                    ],
                    "temperature": llm.get("temperature", 0.1),
                    "max_tokens": llm.get("max_output_tokens", 4000),
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "fundamental_extraction", "strict": True, "schema": schema}},
                }
                if llm.get("enable_web_search"):
                    payload["tools"] = [{"type": llm.get("search_tool", "web_search_preview")}]
                resp = _post(f"{base}/chat/completions", payload,
                             {"Authorization": f"Bearer {key}"}, timeout)
                choice = (resp.get("choices") or [{}])[0]
                if choice.get("finish_reason") == "length":
                    raise LLMError("输出被截断（finish_reason=length），请提高 max_output_tokens 或减少调研范围")
                text = (choice.get("message") or {}).get("content") or ""
            if not text.strip():
                raise LLMError("模型返回空内容")
            obj = json.loads(_extract_json_text(text))
            ok, errs = validate(obj, schema)
            if not ok:
                raise LLMError("schema 校验失败: " + "; ".join(errs[:6]))
            obj.setdefault("meta", {})
            obj["meta"]["mode"] = "api"
            obj["meta"]["model"] = obj["meta"].get("model") or llm.get("model", "unknown")
            return obj, {"status": "ok", "attempt": attempt + 1, "style": style,
                         "raw_usage": resp.get("usage"), "prompt_hash": rendered["hash"]}
        except LLMError as exc:
            last_err = str(exc)
            # schema 不兼容时，第二次尝试退回「纯提示词 + json 模式」
            if attempt == 0 and ("schema" in last_err or "response_format" in last_err or "400" in last_err):
                llm = dict(llm)
                llm["api_style"] = style
            time.sleep(1.5 * (attempt + 1))
        except json.JSONDecodeError as exc:
            last_err = f"JSON 解析失败: {exc}"
            time.sleep(1.0)
    return None, {"status": "error", "message": last_err, "prompt_hash": rendered["hash"]}


def _responses_text(resp: Dict[str, Any]) -> str:
    if resp.get("output_text"):
        return resp["output_text"]
    chunks = []
    for item in resp.get("output") or []:
        for c in item.get("content") or []:
            if c.get("type") in ("output_text", "text"):
                chunks.append(c.get("text", ""))
    return "".join(chunks)


def preflight(settings: Dict[str, Any]) -> Dict[str, Any]:
    """开跑前的自检：是否有 Key、端点是否可达、模型是否已配置。"""
    llm = settings.get("llm", {})
    info = {"base_url": llm.get("base_url"), "model": llm.get("model"),
            "style": llm.get("api_style", "chat"), "has_key": bool(api_key(settings)),
            "web_search": bool(llm.get("enable_web_search"))}
    if not info["has_key"]:
        info["hint"] = "无 Key：可用人工模式，或在环境变量 HARVESTCAST_API_KEY 中配置"
        return info
    try:
        req = urllib.request.Request(f"{info['base_url'].rstrip('/')}/models",
                                     headers={"Authorization": f"Bearer {api_key(settings)}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            info["models_endpoint"] = resp.status
    except Exception as exc:  # noqa: BLE001
        info["models_endpoint"] = f"不可用: {exc}"
    return info
