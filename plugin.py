from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Type

from src.plugin_system import (
    BasePlugin,
    BaseTool,
    ComponentInfo,
    ConfigField,
    PythonDependency,
    ToolParamType,
    get_logger,
    llm_api,
    register_plugin,
)

from maim_message import Seg
from .trace_client import AnimeTraceClient, AnimeTraceError
from src.common.database.database_model import Images


class AnimeTraceTool(BaseTool):
    """Tool wrapper that sends images to AnimeTrace for recognition."""

    name: str = "anime_trace_search"
    description: str = "Recognise anime images via AnimeTrace and return likely works or characters."
    parameters: List[Tuple[str, ToolParamType, str, bool, List[str] | None]] = [
        ("use_reply", ToolParamType.STRING, "Whether to use the referenced message image (true/false)", False, None),
        ("image_index", ToolParamType.INTEGER, "Index of the image inside the selected message (default 1)", False, None),
        ("is_multi", ToolParamType.INTEGER, "Request multiple candidates from AnimeTrace (0 or 1)", False, None),
        ("model", ToolParamType.STRING, "AnimeTrace model name", False, None),
        ("ai_detect", ToolParamType.INTEGER, "Enable AnimeTrace AI detection (1=on, 2=off)", False, None),
        ("min_similarity", ToolParamType.STRING, "Minimum similarity threshold (supports decimal or percent)", False, None),
    ]
    available_for_llm: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        endpoint = self.get_config("anime_trace.endpoint", "https://api.animetrace.com/v1/search")
        timeout = float(self.get_config("anime_trace.request_timeout", 15.0) or 15.0)
        self.client = AnimeTraceClient(endpoint=endpoint, timeout=timeout)
        self.logger = get_logger("animetrace_tool")

    @staticmethod
    def _safe_int(value: Any, default: int, allow: Optional[List[int]] = None) -> int:
        try:
            val = int(value)
            if allow is not None and val not in allow:
                return default
            return val
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return default
            try:
                if txt.endswith("%"):
                    return float(txt[:-1].strip()) / 100.0
                return float(txt)
            except ValueError:
                return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            txt = value.strip().lower()
            if txt in {"true", "1", "yes", "on"}:
                return True
            if txt in {"false", "0", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def _normalize_base64(value: str) -> str:
        cleaned = value.strip()
        if "," in cleaned:
            cleaned = cleaned.split(",", 1)[-1]
        return cleaned.replace("\n", "").replace("\r", "")

    async def _get_base64_from_context(self, *, image_index: int, use_reply: bool) -> str:
        message = self._get_latest_message()
        exclude_ids: set[str] = set()

        # 1 优先使用引用消息
        if use_reply and message and getattr(message, "reply", None):
            reply_candidates = self._extract_images_from_message(message.reply, origin="reply")
            if reply_candidates:
                return await self._pick_image_from_candidates(
                    reply_candidates, image_index, getattr(message.reply.message_info, "message_id", None)
                )

        # 2) 使用当前消息
        if message:
            current_candidates = self._extract_images_from_message(message, origin="current")
            if current_candidates:
                return await self._pick_image_from_candidates(
                    current_candidates, image_index, getattr(message.message_info, "message_id", None)
                )

        # 3) 历史消息兜底
        return await self._select_image_from_history(image_index=image_index, exclude_message_ids=exclude_ids)

    def _get_latest_message(self):
        if not self.chat_stream or not getattr(self.chat_stream, "context", None):
            return None
        try:
            return self.chat_stream.context.get_last_message()
        except Exception:
            return None

    def _extract_images_from_message(self, message, origin: str = "current") -> List[Dict[str, Any]]:
        items: List[str] = []
        if message is None:
            return []
        self._collect_images_from_any(getattr(message, "message_segment", None), items)
        raw_message = getattr(message, "raw_message", None)
        if isinstance(raw_message, str):
            self._collect_images_from_any(raw_message, items)

        unique_items = self._unique_candidates(items)
        message_id = None
        if getattr(message, "message_info", None):
            message_id = getattr(message.message_info, "message_id", None)
        summary = self._build_message_summary(message)
        return [
            {"data": item, "message_id": message_id, "origin": origin, "summary": summary}
            for item in unique_items
        ]

    def _collect_images_from_any(self, obj: Any, out: List[str]) -> None:
        if obj is None:
            return

        if isinstance(obj, Seg):
            seg_type = getattr(obj, "type", "")
            data = getattr(obj, "data", None)
            if seg_type in {"image", "emoji"}:
                self._append_candidate(data, out)
                return
            if seg_type == "seglist" and isinstance(data, list):
                for sub in data:
                    self._collect_images_from_any(sub, out)
                return
            if isinstance(data, (list, dict, str)):
                self._collect_images_from_any(data, out)
            return

        if isinstance(obj, list):
            for item in obj:
                self._collect_images_from_any(item, out)
            return

        if isinstance(obj, dict):
            if "type" in obj and "data" in obj:
                try:
                    seg = Seg.from_dict(obj)
                    self._collect_images_from_any(seg, out)
                except Exception:
                    pass
            for key in ("base64", "url", "file"):
                value = obj.get(key)
                self._append_candidate(value, out)
            for key in ("message_segment", "message", "data", "content", "segments"):
                if key in obj:
                    self._collect_images_from_any(obj[key], out)
            return

        if isinstance(obj, str):
            self._append_candidate(obj, out)

    def _unique_candidates(self, items: List[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for item in items:
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        return ordered

    def _build_message_summary(self, message) -> str:
        info = getattr(message, "message_info", None)
        user_name = ""
        if info and getattr(info, "user_info", None):
            user_name = info.user_info.user_nickname or info.user_info.user_id or ""
        text = getattr(message, "processed_plain_text", "") or getattr(message, "display_message", "") or ""
        summary = text.strip()
        if user_name:
            return f"{user_name}: {summary}" if summary else user_name
        return summary

    def _build_db_message_summary(self, db_msg) -> str:
        try:
            user = db_msg.user_info.user_nickname or db_msg.user_info.user_id or ""
        except Exception:
            user = ""
        text = getattr(db_msg, "display_message", "") or getattr(db_msg, "processed_plain_text", "") or ""
        summary = text.strip()
        if summary and len(summary) > 160:
            summary = summary[:157] + "..."
        if user:
            return f"{user}: {summary}" if summary else user
        return summary

    async def _pick_image_from_candidates(
        self, candidates: List[Dict[str, Any]], image_index: int, message_id: Optional[str]
    ) -> str:
        if not candidates:
            raise AnimeTraceError("未找到可用的图片")
        if image_index <= 0 or image_index > len(candidates):
            raise AnimeTraceError(f"仅找到 {len(candidates)} 张图片，请调整 image_index 参数 (1-{len(candidates)})")
        selected = candidates[image_index - 1]
        base64_data = await self._resolve_candidate_to_base64(selected["data"])
        self.logger.info(
            f"AnimeTrace 使用消息 {message_id or 'unknown'} 中的第 {image_index} 张图片 (origin={selected.get('origin')})"
        )
        return base64_data

    async def _select_image_from_history(
        self, *, image_index: int, exclude_message_ids: Optional[set[str]] = None
    ) -> str:
        if not self.chat_stream:
            raise AnimeTraceError("未找到聊天上下文，请发送图片后重试")

        exclude_message_ids = exclude_message_ids or set()
        history_candidates = self._list_history_candidates(exclude_message_ids)
        if not history_candidates:
            raise AnimeTraceError("未在最近的聊天记录中找到图片，请发送图片后重试")

        if len(history_candidates) == 1:
            candidate = history_candidates[0]
            return await self._pick_image_from_candidates(candidate["candidates"], image_index, candidate["message_id"])

        candidate, suggested_index = await self._choose_candidate_with_llm(history_candidates)
        effective_index = image_index if image_index != 1 else suggested_index
        return await self._pick_image_from_candidates(candidate["candidates"], effective_index, candidate["message_id"])

    def _list_history_candidates(self, exclude_message_ids: set[str]) -> List[Dict[str, Any]]:
        from src.plugin_system.apis import message_api

        lookup_seconds = float(self.get_config("anime_trace.history_lookup_seconds", 3600) or 3600)
        history_limit = int(self.get_config("anime_trace.history_lookup_limit", 20) or 20)
        now = time.time()

        try:
            history = message_api.get_messages_by_time_in_chat_inclusive(
                self.chat_stream.stream_id,
                max(0.0, now - lookup_seconds),
                now,
                limit=history_limit,
                limit_mode="latest",
                filter_mai=False,
                filter_command=False,
            )
        except Exception as err:
            self.logger.error(f"获取历史消息失败: {err}")
            return []

        candidates: List[Dict[str, Any]] = []
        for msg in reversed(history):
            message_id = getattr(msg, "message_id", "") or ""
            if message_id in exclude_message_ids:
                continue
            images = self._extract_images_from_db_message(msg)
            if not images:
                continue
            summary = self._build_db_message_summary(msg)
            candidates.append(
                {
                    "message_id": message_id,
                    "summary": summary,
                    "timestamp": getattr(msg, "time", 0.0),
                    "candidates": [{"data": item, "message_id": message_id, "origin": "history", "summary": summary} for item in images],
                    "image_count": len(images),
                }
            )
        return candidates

    def _extract_images_from_db_message(self, db_msg) -> List[str]:
        images: List[str] = []
        config_str = getattr(db_msg, "additional_config", None)
        if config_str:
            try:
                payload = json.loads(config_str)
                if isinstance(payload, dict):
                    seg_dict = payload.get("message_segment")
                    if isinstance(seg_dict, dict):
                        seg = Seg.from_dict(seg_dict)
                        self._collect_images_from_any(seg, images)
            except Exception:
                pass

        if not images:
            for field in ("display_message", "processed_plain_text"):
                text = getattr(db_msg, field, None)
                if not text:
                    continue
                for match in re.findall(r"\[CQ:image[^\]]+\]", text):
                    images.append(match)
                for match in re.findall(r"(https?://\S+)", text):
                    images.append(match)
                for match in re.findall(r"\[picid:[^\]]+\]", text):
                    images.append(match)
        return self._unique_candidates(images)

    async def _choose_candidate_with_llm(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], int]:
        if not candidates:
            raise AnimeTraceError("未找到候选图片")

        if not self.get_config("anime_trace.use_image_selector_llm", True):
            return candidates[-1], 1

        try:
            models = llm_api.get_available_models()
        except Exception as err:
            self.logger.warning(f"获取模型列表失败，使用默认候选: {err}")
            return candidates[-1], 1

        if not models:
            return candidates[-1], 1

        configured_key = str(
            self.get_config("anime_trace.image_selector_model_key", "")
            or self.get_config("anime_trace.llm_model_key", "")
        ).strip()
        prefer_keys = [configured_key, "plugin_reply", "replyer", "reply", "chat"]
        task = None
        chosen_key = None
        for key in prefer_keys:
            if key and key in models:
                task = models[key]
                chosen_key = key
                break
        if task is None:
            return candidates[-1], 1

        latest_message = self._get_latest_message()
        request_text = ""
        if latest_message:
            request_text = (
                getattr(latest_message, "processed_plain_text", "")
                or getattr(latest_message, "display_message", "")
                or ""
            )

        prompt = self._build_selector_prompt(request_text, candidates)
        try:
            success, reply, _, model_name = await llm_api.generate_with_model(
                prompt, task, request_type="tool.animetrace.select_image"
            )
            if success and isinstance(reply, str):
                parsed = self._parse_selector_response(reply, candidates)
                if parsed:
                    candidate, suggested_index = parsed
                    self.logger.info(
                        f"Image selector LLM({model_name or chosen_key}) 选择消息 {candidate['message_id']} (image_index={suggested_index})"
                    )
                    return candidate, suggested_index
        except Exception as err:
            self.logger.error(f"图片选择模型调用失败: {err}")
        return candidates[-1], 1

    def _build_selector_prompt(self, request_text: str, candidates: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, item in enumerate(candidates, start=1):
            summary = item.get("summary", "") or ""
            lines.append(
                f"{idx}. message_id={item.get('message_id')} | image_count={item.get('image_count')} | 摘要={summary}"
            )
        candidate_block = "\n".join(lines)
        request_block = request_text or "（无）"
        return (
            "你是 AnimeTrace 插件的图片选择助手。用户正在请求识别图片。\n"
            f"用户当前输入：{request_block}\n"
            "以下是最近包含图片的消息列表，请选择最适合识别的消息。\n"
            f"{candidate_block}\n"
            "请只返回一个 JSON，格式为：{\"message_id\": \"消息ID\", \"image_index\": 1}\n"
            "若无法判断，请选择编号最大的消息，并让 image_index=1。仅返回 JSON。"
        )

    def _parse_selector_response(
        self, reply: str, candidates: List[Dict[str, Any]]
    ) -> Optional[Tuple[Dict[str, Any], int]]:
        try:
            match = re.search(r"\{.*\}", reply, re.S)
            if not match:
                return None
            data = json.loads(match.group(0))
            message_id = data.get("message_id")
            image_index = int(data.get("image_index", 1))
            if not message_id:
                return None
            for candidate in candidates:
                if candidate.get("message_id") == message_id:
                    if image_index <= 0:
                        image_index = 1
                    return candidate, image_index
        except Exception:
            return None
        return None

    def _append_candidate(self, value: Any, out: List[str]) -> None:
        if not isinstance(value, str):
            return
        candidate = value.strip()
        if not candidate:
            return
        if self._is_likely_image_string(candidate):
            out.append(candidate)

    def _is_likely_image_string(self, value: str) -> bool:
        if value.startswith("http://") or value.startswith("https://"):
            return True
        if value.startswith("base64://") or value.startswith("data:"):
            return True
        if value.startswith("[CQ:image"):
            return True
        if "[picid:" in value:
            return True
        if self._looks_like_base64(value):
            return True
        return False

    def _looks_like_base64(self, value: str) -> bool:
        if len(value) < 80:
            return False
        try:
            base64.b64decode(value, validate=True)
            return True
        except Exception:
            return False

    async def _resolve_candidate_to_base64(self, candidate: str) -> str:
        source = candidate.strip()
        if not source:
            raise AnimeTraceError("Image data is empty; please provide a valid image.")

        if source.startswith("base64://"):
            return self._normalize_base64(source[len("base64://") :])
        if source.startswith("data:"):
            return self._normalize_base64(source.split(",", 1)[-1])

        if source.startswith("[CQ:image"):
            base64_match = re.search(r"base64=([^,\\]]+)", source)
            if base64_match:
                return self._normalize_base64(base64_match.group(1))
            url_match = re.search(r"url=([^,\\]]+)", source)
            if url_match:
                remote_base64 = await self.client.url_to_base64(url_match.group(1))
                return self._normalize_base64(remote_base64)
            file_match = re.search(r"file=([^,\\]]+)", source)
            if file_match:
                file_value = file_match.group(1)
                if file_value.startswith("base64://"):
                    return self._normalize_base64(file_value[len("base64://") :])
                if file_value.startswith("http://") or file_value.startswith("https://"):
                    remote_base64 = await self.client.url_to_base64(file_value)
                    return self._normalize_base64(remote_base64)

        picid_match = re.search(r"\[picid:([^\]]+)\]", source)
        if picid_match:
            return await self._picid_to_base64(picid_match.group(1))

        if source.startswith("http://") or source.startswith("https://"):
            remote_base64 = await self.client.url_to_base64(source)
            return self._normalize_base64(remote_base64)

        if self._looks_like_base64(source):
            return self._normalize_base64(source)

        raise AnimeTraceError("Unable to resolve image data; please make sure the image is accessible.")

    async def _picid_to_base64(self, image_id: str) -> str:
        try:
            image = Images.get_or_none(Images.image_id == image_id)
            if not image:
                raise AnimeTraceError(f"Image record not found for picid: {image_id}")
            path = getattr(image, "path", "") or ""
            if not path or not os.path.exists(path):
                raise AnimeTraceError(f"Image file missing for picid: {image_id}")
            with open(path, "rb") as fh:
                data = fh.read()
            return base64.b64encode(data).decode("utf-8")
        except AnimeTraceError:
            raise
        except Exception as exc:
            raise AnimeTraceError(f"Failed to load image for picid {image_id}: {exc}")

    async def execute(self, function_args: Dict[str, Any]) -> Dict[str, str]:
        use_reply = self._safe_bool(function_args.get("use_reply", False), False)
        image_index = self._safe_int(function_args.get("image_index", 1), 1)
        if image_index <= 0:
            image_index = 1

        is_multi_default = self.get_config("anime_trace.is_multi", 1)
        ai_detect_default = self.get_config("anime_trace.ai_detect", 1)
        model_default = self.get_config("anime_trace.model", "animetrace_high_beta")
        min_similarity_default = self.get_config("anime_trace.min_similarity", 0.0)

        is_multi = self._safe_int(function_args.get("is_multi", is_multi_default), int(is_multi_default or 1), [0, 1])
        ai_detect = self._safe_int(function_args.get("ai_detect", ai_detect_default), int(ai_detect_default or 1), [1, 2])
        model = str(function_args.get("model", model_default) or "").strip() or "animetrace_high_beta"
        min_similarity = self._safe_float(
            function_args.get("min_similarity", min_similarity_default), float(min_similarity_default or 0.0)
        )

        try:
            chosen_base64 = await self._get_base64_from_context(image_index=image_index, use_reply=use_reply)
        except AnimeTraceError as err:
            return {"name": self.name, "content": str(err)}

        try:
            response = await self.client.search(
                base64_data=chosen_base64,
                model=model,
                is_multi=is_multi,
                ai_detect=ai_detect,
            )
        except AnimeTraceError as err:
            return {"name": self.name, "content": f"识别失败: {err}"}

        threshold = AnimeTraceClient._norm_similarity(AnimeTraceClient._as_float(min_similarity)) or 0.0
        summary = AnimeTraceClient.format_response_text(response, min_similarity=min_similarity)
        candidate_lines = self._build_candidate_lines(response, threshold)

        content_parts = [summary] if summary else ["未获得识别结果"]
        if candidate_lines:
            content_parts.append("\n".join(candidate_lines))

        trace_id = response.get("trace_id")
        if trace_id:
            content_parts.append(f"trace_id: {trace_id}")

        content = "\n".join(content_parts)

        if self.get_config("anime_trace.use_llm", False):
            llm_text = await self._maybe_llm_summarize(content)
            if llm_text:
                content = llm_text

        return {"name": self.name, "content": content}

    def _build_candidate_lines(self, resp: Dict[str, Any], threshold: float) -> List[str]:
        data = resp.get("data")
        candidates: List[Dict[str, Any]] = []

        if isinstance(data, list):
            candidates = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            for key in ("results", "result"):
                candidate_list = data.get(key)
                if isinstance(candidate_list, list):
                    candidates = [item for item in candidate_list if isinstance(item, dict)]
                    break

        lines: List[str] = []
        for idx, item in enumerate(candidates[:5], start=1):
            line = self._format_candidate(item, idx, threshold)
            if line:
                lines.append(line)
        return lines

    def _format_candidate(self, item: Dict[str, Any], idx: int, threshold: float) -> Optional[str]:
        characters = item.get("character")
        if isinstance(characters, list) and characters:
            top = []
            for ch in characters[:3]:
                if not isinstance(ch, dict):
                    continue
                work = (ch.get("work") or "").strip()
                name = (ch.get("character") or "").strip()
                if work and name:
                    top.append(f"{work}:{name}")
                elif name:
                    top.append(name)
                elif work:
                    top.append(work)
            if top:
                return f"{idx}. 角色候选：{'；'.join(top)}"

        title = (
            item.get("title")
            or item.get("name")
            or item.get("subject")
            or item.get("source")
            or item.get("label")
            or ""
        ).strip()
        if not title:
            return None

        similarity_raw = (
            item.get("similarity")
            or item.get("score")
            or item.get("sim")
            or item.get("confidence")
        )
        similarity = AnimeTraceClient._norm_similarity(AnimeTraceClient._as_float(similarity_raw))
        if threshold and similarity is not None and similarity < threshold:
            return None

        extras: List[str] = []
        for key in ("episode", "part", "chapter", "time"):
            value = item.get(key)
            if value:
                extras.append(f"{key}:{value}")
        if item.get("site"):
            extras.append(str(item.get("site")))
        if item.get("url"):
            extras.append(str(item.get("url")))

        line = f"{idx}. {title}"
        if similarity is not None:
            line += f"（相似度 {similarity:.1f}%）"
        if extras:
            line += " - " + " ".join(extras)
        return line

    async def _maybe_llm_summarize(self, text: str) -> Optional[str]:
        try:
            models = llm_api.get_available_models()
        except Exception:
            return None

        if not models:
            return None

        configured_key = str(self.get_config("anime_trace.llm_model_key", "plugin_reply") or "").strip()
        prefer_keys = [key for key in [configured_key, "plugin_reply", "replyer", "reply", "chat"] if key]

        task = None
        for key in prefer_keys:
            if key in models:
                task = models[key]
                break

        if task is None:
            try:
                task = next(iter(models.values()))
            except StopIteration:
                return None

        prompt = (
            "你是动漫识别助手。以下是 AnimeTrace 返回的识别摘要，请用简洁中文概括，突出最可能的作品或角色，"
            "如包含相似度则保留关键信息，控制在三行以内：\n\n"
            f"{text}"
        )

        try:
            success, result_text, _, model_name = await llm_api.generate_with_model(
                prompt, task, request_type="tool.animetrace.summarize"
            )
            if success and isinstance(result_text, str) and result_text.strip():
                self.logger.info(f"LLM summarize via {model_name}")
                return result_text.strip()
        except Exception as err:
            self.logger.error(f"LLM 总结失败: {err}")

        return None


@register_plugin
class AnimeTracePlugin(BasePlugin):
    """AnimeTrace 图片识别工具插件。"""

    plugin_name: str = "animetrace_plugin"
    enable_plugin: bool = True
    dependencies: List[str] = []
    python_dependencies: List[PythonDependency] = [
        PythonDependency(
            package_name="httpx",
            version=">=0.24.0",
            optional=True,
            description="HTTP 客户端，用于向 AnimeTrace 发送请求",
        ),
    ]
    config_file_name: str = "config.toml"

    config_section_descriptions = {
        "plugin": "插件基础配置",
        "anime_trace": "AnimeTrace 接口参数与工具返回设置",
    }

    config_schema: Dict[str, Dict[str, ConfigField]] = {
    "plugin": {
        "config_version": ConfigField(
            type=str,
            default="2.2.0",
            description="Config file version",
        ),
        "enabled": ConfigField(
            type=bool,
            default=True,
            description="Enable plugin",
        ),
    },
    "anime_trace": {
        "endpoint": ConfigField(
            type=str,
            default="https://api.animetrace.com/v1/search",
            description="AnimeTrace API endpoint",
        ),
        "model": ConfigField(
            type=str,
            default="animetrace_high_beta",
            description="Default AnimeTrace model",
        ),
        "is_multi": ConfigField(
            type=int,
            default=1,
            description="Return multiple candidates (0/1)",
        ),
        "ai_detect": ConfigField(
            type=int,
            default=1,
            description="Enable AnimeTrace AI detection (1/2)",
        ),
        "request_timeout": ConfigField(
            type=float,
            default=15.0,
            description="HTTP request timeout in seconds",
        ),
        "min_similarity": ConfigField(
            type=float,
            default=0.0,
            description="Minimum similarity threshold",
        ),
        "history_lookup_seconds": ConfigField(
            type=float,
            default=3600.0,
            description="Lookback window for history search (seconds)",
        ),
        "history_lookup_limit": ConfigField(
            type=int,
            default=20,
            description="Maximum number of history messages to scan",
        ),
        "use_image_selector_llm": ConfigField(
            type=bool,
            default=True,
            description="Use LLM to choose history images",
        ),
        "image_selector_model_key": ConfigField(
            type=str,
            default="plugin_reply",
            description="Model key for history image selection",
        ),
        "use_llm": ConfigField(
            type=bool,
            default=False,
            description="Summarize AnimeTrace result with LLM",
        ),
        "llm_model_key": ConfigField(
            type=str,
            default="plugin_reply",
            description="Model key for final response summary",
        ),
    },
}

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        return [
            (AnimeTraceTool.get_tool_info(), AnimeTraceTool),
        ]
