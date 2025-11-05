from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple, Type

from src.plugin_system import (
    BaseEventHandler,
    BasePlugin,
    ComponentInfo,
    ConfigField,
    EventType,
    MaiMessages,
    PythonDependency,
    get_logger,
    register_plugin,
    llm_api,
)

from .trace_client import AnimeTraceClient, AnimeTraceError


class AnimeTraceOnMessage(BaseEventHandler):
    """
    事件处理器：接收图片消息并调用 AnimeTrace 进行识别，返回识别文本结果。

    行为策略可通过插件配置控制：支持自动识别图片或关键字触发。
    """

    event_type = EventType.ON_MESSAGE
    handler_name = "animetrace_on_message"
    handler_description = "接收图片并调用 AnimeTrace 返回识别结果"
    weight = 10
    intercept_message = False
    _logger = get_logger("animetrace_plugin")

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], None, Optional[MaiMessages]]:
        if not message:
            return True, True, None, None, None

        stream_id = message.stream_id or ""
        if not stream_id:
            return True, True, None, None, None

        # 读取配置
        auto_on_image: bool = bool(self.get_config("anime_trace.auto_on_image", True))
        trigger_keywords: List[str] = list(
            self.get_config(
                "anime_trace.trigger_keywords", ["/trace", "动漫识别", "以图搜番"]
            )
        )  # type: ignore
        max_images: int = int(self.get_config("anime_trace.max_images", 1) or 1)
        min_similarity: float = float(
            self.get_config("anime_trace.min_similarity", 0.0) or 0.0
        )
        input_preference: str = str(
            self.get_config("anime_trace.input_preference", "prefer_base64")
        )

        text = (message.plain_text or "").strip()

        # 是否触发识别
        should_trigger = False
        if auto_on_image:
            should_trigger = True
        else:
            for kw in trigger_keywords:
                if kw and kw in text:
                    should_trigger = True
                    break

        if not should_trigger:
            return True, True, None, None, None

        # 提取图片段（支持 image/emoji 以及 Napcat data.file、嵌套 seglist/forward/reply 的通用解析）
        def _collect_images_from_any(obj: Any, out: List[str]) -> None:
            if obj is None:
                return
            # Seg 对象
            if hasattr(obj, "type") and hasattr(obj, "data"):
                s_type = getattr(obj, "type", "")
                s_data = getattr(obj, "data", None)
                if s_type in ("image", "emoji") and isinstance(s_data, str):
                    out.append(s_data)
                    return
                # 递归处理列表/字典形态
                _collect_images_from_any(s_data, out)
                return
            # list 容器
            if isinstance(obj, list):
                for item in obj:
                    _collect_images_from_any(item, out)
                return
            # dict 容器（Napcat 风格）
            if isinstance(obj, dict):
                # 常见字段：file/url/base64、message(嵌套消息列表)、content、data
                file_val = obj.get("file")
                if isinstance(file_val, str):
                    out.append(file_val)
                for k in ("message", "content", "data"):
                    if k in obj:
                        _collect_images_from_any(obj[k], out)
                return

        images: List[str] = []
        _collect_images_from_any(message.message_segments or [], images)
        # 去重并截断
        seen: set[str] = set()
        uniq_images: List[str] = []
        for v in images:
            if v not in seen:
                seen.add(v)
                uniq_images.append(v)
            if len(uniq_images) >= max_images:
                break
        images = uniq_images

        if not images:
            return True, True, None, None, None

        # 回执提示与日志
        await self.send_text(stream_id, "正在识别图片(AnimeTrace)…")
        try:
            self._logger.info(
                f"Start trace: count={len(images)} pref={input_preference} model={self.get_config('anime_trace.model', 'animetrace_high_beta')}"
            )
        except Exception:
            pass

        # 客户端配置
        endpoint: str = str(
            self.get_config(
                "anime_trace.endpoint", "https://api.animetrace.com/v1/search"
            )
        )
        model: str = str(self.get_config("anime_trace.model", "animetrace_high_beta"))
        is_multi: int = int(self.get_config("anime_trace.is_multi", 1) or 1)
        ai_detect: int = int(self.get_config("anime_trace.ai_detect", 1) or 1)
        request_timeout: float = float(
            self.get_config("anime_trace.request_timeout", 15.0) or 15.0
        )

        client = AnimeTraceClient(endpoint=endpoint, timeout=request_timeout)

        lines: List[str] = []
        for idx, img in enumerate(images, start=1):
            try:
                # 判断是否是 URL 或 base64，根据 input_preference 统一路径
                if img.startswith("http://") or img.startswith("https://"):
                    try:
                        self._logger.info(f"Image #{idx} url={img}")
                    except Exception:
                        pass
                    if input_preference == "prefer_base64":
                        # 尝试将 URL 下载并转 Base64，再以 base64 字段上传
                        try:
                            b64 = await client.url_to_base64(img)
                            try:
                                self._logger.info(
                                    f"Image #{idx} url-to-base64 len={len(b64)} head={b64[:24]}"
                                )
                            except Exception:
                                pass
                            resp = await client.search(
                                base64_data=b64,
                                model=model,
                                is_multi=is_multi,
                                ai_detect=ai_detect,
                            )
                            self._logger.info("Trace via base64 (from url)")
                        except Exception:
                            # 降级：直接用 URL 请求
                            resp = await client.search(
                                url=img,
                                model=model,
                                is_multi=is_multi,
                                ai_detect=ai_detect,
                            )
                            self._logger.info("Trace via url (fallback)")
                    else:
                        resp = await client.search(
                            url=img, model=model, is_multi=is_multi, ai_detect=ai_detect
                        )
                        self._logger.info("Trace via url")
                else:
                    # 粗略校验是否为base64，并规范为纯b64（去掉 data:* 前缀）
                    raw_part = img.split(",")[-1]
                    try:
                        base64.b64decode(raw_part, validate=True)
                        is_b64 = True
                    except Exception:
                        is_b64 = False

                    if not is_b64:
                        # 无法识别的内容，跳过
                        continue
                    try:
                        self._logger.info(
                            f"Image #{idx} base64 len={len(raw_part)} head={raw_part[:24]}"
                        )
                    except Exception:
                        pass
                    try:
                        resp = await client.search(
                            base64_data=raw_part,
                            model=model,
                            is_multi=is_multi,
                            ai_detect=ai_detect,
                        )
                        self._logger.info("Trace via base64")
                    except AnimeTraceError as e_base64:
                        # 若 base64 路径 4xx，则回退为文件上传（更贴合部分后端实现）
                        # 注意：仅在 httpx 可用时 file_bytes 才生效；client 内部会抛错提示
                        try:
                            import base64 as _b64

                            file_bytes = _b64.b64decode(raw_part)
                            self._logger.info(
                                f"Fallback to file upload (#{idx}) bytes={len(file_bytes)}"
                            )
                            resp = await client.search(
                                file_bytes=file_bytes,
                                model=model,
                                is_multi=is_multi,
                                ai_detect=ai_detect,
                            )
                            self._logger.info("Trace via file (fallback from base64)")
                        except Exception:
                            raise e_base64

                # 响应结构简要日志
                try:
                    d = resp.get("data") if isinstance(resp, dict) else None
                    d_type = (
                        "list"
                        if isinstance(d, list)
                        else "dict"
                        if isinstance(d, dict)
                        else "none"
                    )
                    d_len = (
                        len(d)
                        if isinstance(d, list)
                        else (len(d.keys()) if isinstance(d, dict) else 0)
                    )
                    self._logger.info(
                        f"Resp keys={list(resp.keys())[:6] if isinstance(resp, dict) else 'n/a'} data_type={d_type} data_len={d_len}"
                    )
                except Exception:
                    pass

                # 将响应转换为简明文本
                text_line = AnimeTraceClient.format_response_text(
                    resp, min_similarity=min_similarity
                )
                if len(images) > 1:
                    text_line = f"[#{idx}] {text_line}"
                lines.append(text_line)
            except AnimeTraceError as e:
                lines.append(f"识别失败: {e}")
            except Exception as e:  # 兜底
                lines.append(f"识别异常: {e}")

        if not lines:
            lines = ["未获得识别结果"]

        reply_text = "\n".join(lines)

        # === LLM 处理：将识别结果传递给 LLM，再将 LLM 生成内容回复 ===
        use_llm: bool = bool(self.get_config("anime_trace.use_llm", True))
        if use_llm:
            try:
                models = llm_api.get_available_models()
                # 强制优先使用 MaiBot 的 replyer 模型（plugin_reply / replyer / reply），可由配置覆盖键名
                configured_key = str(
                    self.get_config("anime_trace.llm_model_key", "plugin_reply")
                ).strip()
                prefer_keys = [
                    k
                    for k in [
                        configured_key,
                        "plugin_reply",
                        "replyer",
                        "reply",
                        "chat",
                    ]
                    if k
                ]
                task = None
                for key in prefer_keys:
                    if key in models:
                        task = models[key]
                        break
                if not task and models:
                    self._logger.warning(
                        "未找到指定的 replyer 模型键，使用第一个可用模型兜底"
                    )
                    task = next(iter(models.values()))

                prompt_header = (
                    "你是动漫识别助手。以下是识别结果摘要，请用简洁中文生成可读回复，"
                    "若包含角色TopN则简要解释；若包含相似度则给出最可能项，两三行内完成：\n\n"
                )
                prompt = prompt_header + reply_text

                if task:
                    success, gen, _, model_name = await llm_api.generate_with_model(
                        prompt, task, request_type="plugin.animetrace.summarize"
                    )
                    self._logger.info(
                        f"LLM generate success={success} model={model_name} len={len(gen) if isinstance(gen, str) else 0}"
                    )
                    await self.send_text(stream_id, gen if success else reply_text)
                else:
                    self._logger.warning("未找到可用LLM模型，直接返回识别摘要")
                    await self.send_text(stream_id, reply_text)
            except Exception as e:
                self._logger.error(f"LLM 处理异常: {e}")
                await self.send_text(stream_id, reply_text)
        else:
            await self.send_text(stream_id, reply_text)
        return True, True, None, None, None


@register_plugin
class AnimeTracePlugin(BasePlugin):
    """AnimeTrace 图片识别插件"""

    plugin_name: str = "animetrace_plugin"
    enable_plugin: bool = True
    dependencies: List[str] = []
    python_dependencies: List[PythonDependency] = [
        # HTTP 客户端依赖（宿主无自动安装，保留声明以便提示）
        PythonDependency(
            package_name="httpx",
            version=">=0.24.0",
            optional=True,
            description="HTTP 客户端",
        ),
    ]
    config_file_name: str = "config.toml"

    config_section_descriptions = {
        "plugin": "插件基本信息",
        "anime_trace": "AnimeTrace 接口与识别控制",
    }

    config_schema: Dict[str, Dict[str, ConfigField]] = {
        "plugin": {
            "config_version": ConfigField(
                type=str, default="1.0.0", description="配置文件版本"
            ),
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
        },
        "anime_trace": {
            "endpoint": ConfigField(
                type=str,
                default="https://api.animetrace.com/v1/search",
                description="API 地址",
            ),
            "model": ConfigField(
                type=str, default="animetrace_high_beta", description="识别模型"
            ),
            "is_multi": ConfigField(
                type=int, default=1, description="显示多个结果(1/0)"
            ),
            "ai_detect": ConfigField(type=int, default=1, description="AI图检测(1/0)"),
            "request_timeout": ConfigField(
                type=float, default=20.0, description="请求超时(秒)"
            ),
            "auto_on_image": ConfigField(
                type=bool, default=False, description="收到图片自动识别"
            ),
            "trigger_keywords": ConfigField(
                type=list,
                default=["/trace", "动漫识别", "以图搜番"],
                description="关键词触发",
            ),
            "max_images": ConfigField(
                type=int, default=1, description="一次处理的最大图片数"
            ),
            "min_similarity": ConfigField(
                type=float, default=0.0, description="最小相似度阈值(0-100或0-1)"
            ),
            "input_preference": ConfigField(
                type=str,
                default="prefer_base64",
                description="输入优先级: auto/prefer_url/prefer_base64",
            ),
            "use_llm": ConfigField(
                type=bool, default=True, description="是否将结果交由LLM生成回复"
            ),
            "llm_model_key": ConfigField(
                type=str,
                default="plugin_reply",
                description="指定 LLM 模型键（默认 replyer）",
            ),
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        return [
            (AnimeTraceOnMessage.get_handler_info(), AnimeTraceOnMessage),
        ]
