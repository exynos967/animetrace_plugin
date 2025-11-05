from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - 作为可选依赖
    httpx = None  # type: ignore

from src.common.logger import get_logger


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
        self.logger = get_logger("animetrace_client")

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
        # 请求体（JSON 优先），数值字段保持原始类型
        payload: Dict[str, Any] = {
            "model": model,
            "is_multi": is_multi,
            "ai_detect": ai_detect,
        }
        files = None

        if url:
            payload["url"] = url
        elif base64_data:
            # 统一为纯Base64（移除可能存在的 data:* 前缀）
            payload["base64"] = base64_data.split(",")[-1].strip()
        elif file_bytes is not None:
            # httpx 支持更方便的 multipart 文件上传
            files = {"file": ("image.png", file_bytes, "image/png")}
        else:
            raise AnimeTraceError("必须提供 url / base64 / file 之一")

        # 记录请求体（脱敏/截断大字段）
        try:
            payload_log = self._build_log_payload(payload, files, file_bytes)
            self.logger.info(f"[AnimeTrace] Request body(JSON default): {payload_log}")
        except Exception:
            pass

        if httpx is None:
            raise AnimeTraceError(
                "缺少 httpx 依赖，请安装 httpx 以使用 AnimeTrace 请求"
            )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                if files:
                    resp = await client.post(self.endpoint, data=payload, files=files)
                else:
                    resp = await client.post(self.endpoint, json=payload)
            except httpx.HTTPError as e:  # type: ignore
                raise AnimeTraceError(f"网络异常: {e}")

            # info 日志：状态码与内容长度
            try:
                self.logger.info(
                    f"[AnimeTrace] POST {self.endpoint} status={resp.status_code} content-length={resp.headers.get('content-length', '-')}"
                )
            except Exception:
                pass

            if resp.status_code >= 400:
                err_excerpt = ""
                try:
                    txt = resp.text[:200]
                    try:
                        j = resp.json()
                        msg = j.get("message") or j.get("msg") or ""
                        code = j.get("code")
                        if msg:
                            err_excerpt = f" code={code} msg={msg}"
                        else:
                            err_excerpt = f" body={txt}"
                    except Exception:
                        err_excerpt = f" body={txt}"
                except Exception:
                    pass
                raise AnimeTraceError(
                    f"HTTP 错误{err_excerpt}", status=resp.status_code
                )

            try:
                return resp.json()  # type: ignore
            except Exception:
                raise AnimeTraceError("响应解析失败")

    def _build_log_payload(
        self, payload: Dict[str, Any], files, file_bytes: Optional[bytes]
    ) -> Dict[str, Any]:
        """构建用于日志的请求体，截断大字段，避免输出整段Base64。"""
        red: Dict[str, Any] = dict(payload)
        b64 = red.get("base64")
        if isinstance(b64, str):
            red["base64"] = f"<base64 len={len(b64)} head={b64[:24]}>"
        if files is not None:
            length = len(file_bytes) if file_bytes is not None else 0
            red["file"] = f"<file bytes={length}>"
        return red

    # --------- 下载工具（要求 httpx） ---------
    async def download_bytes(self, url: str) -> bytes:
        """下载图片字节，用于 URL → base64 或文件上传路径。"""
        if httpx is None:
            raise AnimeTraceError("缺少 httpx 依赖，无法下载图片")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(url)
                self.logger.info(
                    f"[AnimeTrace] GET {url} status={resp.status_code} len={len(resp.content)}"
                )
                if resp.status_code >= 400:
                    raise AnimeTraceError("下载失败", status=resp.status_code)
                return resp.content
            except httpx.HTTPError as e:  # type: ignore
                raise AnimeTraceError(f"下载异常: {e}")

    async def url_to_base64(self, url: str) -> str:
        data = await self.download_bytes(url)
        import base64

        return base64.b64encode(data).decode("ascii")

    # --------- 文本格式化工具 ---------
    @staticmethod
    def _as_float(val: Any) -> Optional[float]:
        try:
            if isinstance(val, str):
                import re

                m = re.search(r"[-+]?[0-9]*\.?[0-9]+", val)
                if m:
                    return float(m.group(0))
                return None
            return float(val)
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
        # 优先处理通用 code/message 语义
        try:
            code = resp.get("code")
            if isinstance(code, int) and code not in (0, 200, 17720):
                zh_msg = resp.get("zh_message")
                msg = resp.get("message")
                return str(zh_msg or msg or f"错误代码: {code}")
        except Exception:
            pass
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

        # 动态选择摘要策略
        def parse_item(item: dict[str, Any]) -> tuple[float, str]:
            # 1) 角色结果（animetrace_demo 中的 data[0].character 列表）
            if isinstance(item.get("character"), list) and item.get("character"):
                chars = item.get("character")
                top = []
                for ch in chars[:3]:
                    try:
                        w = ch.get("work")
                        c = ch.get("character")
                        if w or c:
                            top.append(f"{w or ''}:{c or ''}".strip(":"))
                    except Exception:
                        continue
                summary = (
                    "角色Top" + str(len(top)) + ": " + "; ".join(top)
                    if top
                    else "角色信息返回"
                )
                # 无相似度时返回 0 作为排序
                return 0.0, summary

            # 2) 通用相似度/标题结果
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
