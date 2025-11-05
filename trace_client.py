from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - 作为可选依赖
    httpx = None  # type: ignore

from urllib import request as urllib_request
from urllib.error import URLError, HTTPError


@dataclass
class AnimeTraceError(Exception):
    message: str
    status: int | None = None

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.message}{f' (status {self.status})' if self.status else ''}"


class AnimeTraceClient:
    """AnimeTrace API 客户端。

    仅封装必要接口：/v1/search
    支持 url/base64/file 三种输入（优先使用 url/base64）。
    """

    def __init__(self, endpoint: str, timeout: float = 15.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    async def search(
        self,
        *,
        url: Optional[str] = None,
        base64_data: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        model: str = "animetrace_high_beta",
        is_multi: int = 1,
        ai_detect: int = 1,
    ) -> Dict[str, Any]:
        payload = {"model": model, "is_multi": is_multi, "ai_detect": ai_detect}
        files = None

        if url:
            payload["url"] = url
        elif base64_data:
            payload["base64"] = base64_data
        elif file_bytes is not None:
            # httpx 支持更方便的 multipart 文件上传
            files = {"file": ("image.png", file_bytes, "image/png")}
        else:
            raise AnimeTraceError("必须提供 url / base64 / file 之一")

        if httpx is not None:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    if files:
                        resp = await client.post(
                            self.endpoint, data=payload, files=files
                        )
                    else:
                        # 对于 url/base64 统一用 JSON 以减少兼容问题
                        resp = await client.post(self.endpoint, json=payload)
                except httpx.HTTPError as e:  # type: ignore
                    raise AnimeTraceError(f"网络异常: {e}")

                if resp.status_code >= 400:
                    raise AnimeTraceError("HTTP 错误", status=resp.status_code)
                try:
                    return resp.json()  # type: ignore
                except Exception:
                    raise AnimeTraceError("响应解析失败")

        # 兼容：无 httpx 时使用内置 urllib 执行 JSON 请求（仅支持 url/base64）
        if files is not None:
            raise AnimeTraceError("在缺少 httpx 时不支持文件上传，请改用 url/base64")

        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.endpoint, data=data, headers={"Content-Type": "application/json"}
        )

        def _do() -> Dict[str, Any]:
            try:
                with urllib_request.urlopen(req, timeout=self.timeout) as r:  # nosec - 外部 URL 由配置控制
                    content = r.read().decode("utf-8")
                return json.loads(content)
            except HTTPError as e:
                raise AnimeTraceError("HTTP 错误", status=e.code)
            except URLError as e:  # pragma: no cover
                raise AnimeTraceError(f"网络异常: {e.reason}")
            except Exception:
                raise AnimeTraceError("响应解析失败")

        return await asyncio.to_thread(_do)

    # --------- 文本格式化工具 ---------
    @staticmethod
    def _as_float(val: Any) -> Optional[float]:
        try:
            f = float(val)
            return f
        except Exception:
            return None

    @staticmethod
    def _norm_similarity(sim: Optional[float]) -> Optional[float]:
        if sim is None:
            return None
        # 可能返回 0-1 或 0-100，将其规范到百分比
        if sim <= 1.0:
            return sim * 100.0
        return sim

    @staticmethod
    def format_response_text(
        resp: Dict[str, Any], *, min_similarity: float = 0.0
    ) -> str:
        """将 AnimeTrace 响应压缩为一行可读文本。

        兼容未知字段结构：优先在 data/results 列表中查找；否则回退到整体 JSON 文本（截断）。
        """
        # 归一化阈值到百分比
        thr = (
            AnimeTraceClient._norm_similarity(
                AnimeTraceClient._as_float(min_similarity)
            )
            or 0.0
        )

        candidates: list[dict[str, Any]] = []
        data = resp.get("data")
        if isinstance(data, list):
            candidates = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            if isinstance(data.get("results"), list):
                candidates = [x for x in data.get("results", []) if isinstance(x, dict)]
            elif isinstance(data.get("result"), list):
                candidates = [x for x in data.get("result", []) if isinstance(x, dict)]

        if not candidates and isinstance(resp.get("results"), list):
            candidates = [x for x in resp.get("results", []) if isinstance(x, dict)]

        # 选择最优候选
        def parse_item(item: dict[str, Any]) -> tuple[float, str]:
            title = (
                item.get("title")
                or item.get("name")
                or item.get("subject")
                or item.get("source")
                or "(未知)"
            )
            similarity = (
                AnimeTraceClient._as_float(item.get("similarity"))
                or AnimeTraceClient._as_float(item.get("score"))
                or AnimeTraceClient._as_float(item.get("sim"))
                or 0.0
            )
            sim_pct = AnimeTraceClient._norm_similarity(similarity) or 0.0
            extra = []
            for key in ("episode", "part", "chapter", "time"):
                if item.get(key):
                    extra.append(f"{key}:{item.get(key)}")
            if item.get("url"):
                extra.append(str(item.get("url")))
            if item.get("site"):
                extra.append(str(item.get("site")))
            summary = f"{title}  相似度:{sim_pct:.1f}%" + (
                f"  ({' '.join(extra)})" if extra else ""
            )
            return sim_pct, summary

        if candidates:
            best_text = None
            best_score = -1.0
            for it in candidates:
                score, text = parse_item(it)
                if score >= thr and score > best_score:
                    best_score, best_text = score, text
            if best_text:
                return best_text

        # 回退：取 message / msg 字段或整体 JSON 截断
        for k in ("message", "msg"):
            if resp.get(k):
                return str(resp[k])
        try:
            js = json.dumps(resp, ensure_ascii=False)[:800]
            return js
        except Exception:  # pragma: no cover
            return str(resp)
