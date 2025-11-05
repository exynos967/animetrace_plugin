from __future__ import annotations

import base64
from typing import Dict, List, Optional, Tuple, Type

from src.plugin_system import (
    BaseEventHandler,
    BasePlugin,
    ComponentInfo,
    ConfigField,
    EventType,
    MaiMessages,
    PythonDependency,
    register_plugin,
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

        # 提取图片段（含 image/emoji 两种可能承载 base64 图像数据的段）
        images: List[str] = []
        for seg in message.message_segments or []:
            seg_type = getattr(seg, "type", "")
            seg_data = getattr(seg, "data", None)
            if seg_type in ("image", "emoji") and isinstance(seg_data, str):
                images.append(seg_data)
                if len(images) >= max_images:
                    break

        if not images:
            return True, True, None, None, None

        # 回执提示
        await self.send_text(stream_id, "正在识别图片(AnimeTrace)…")

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
                # 判断是否是 URL 或 base64
                if img.startswith("http://") or img.startswith("https://"):
                    resp = await client.search(
                        url=img, model=model, is_multi=is_multi, ai_detect=ai_detect
                    )
                else:
                    # 粗略校验是否为base64
                    try:
                        base64.b64decode(img.split(",")[-1], validate=True)
                        is_b64 = True
                    except Exception:
                        is_b64 = False

                    if not is_b64:
                        # 无法识别的内容，跳过
                        continue
                    resp = await client.search(
                        base64_data=img,
                        model=model,
                        is_multi=is_multi,
                        ai_detect=ai_detect,
                    )

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
                type=float, default=15.0, description="请求超时(秒)"
            ),
            "auto_on_image": ConfigField(
                type=bool, default=True, description="收到图片自动识别"
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
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        return [
            (AnimeTraceOnMessage.get_handler_info(), AnimeTraceOnMessage),
        ]
