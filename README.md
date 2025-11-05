# AnimeTrace 图片识别插件（MaiBot）

使用 AnimeTrace 服务对 QQ 图片进行识别的 MaiBot 插件。本插件以「Tool」组件形式提供 anime_trace_search 工具，支持从回复/当前/历史消息中自动选取图片，调用 AnimeTrace 并返回可读的结果摘要与候选列表。

## 功能特性
- Tool 组件：`anime_trace_search`，由 LLM 按需调用，也可在插件内直接调用
- 图片来源自动解析：支持 Seg（image/emoji/seglist）、CQ 码 `[CQ:image ...]`、`[picid:...]`、URL、Base64
- 历史消息回溯：可配置时间窗口与条数，并可选用 LLM 辅助选择最佳图片
- 结果摘要：优先显示角色 TopN 或最高相似度候选（相似度统一为百分比），支持最小相似度过滤
- 健壮性：HTTP 重试/退避、URL→Base64 LRU 缓存、Base64 日志脱敏
- 可选 LLM 总结：对长候选结果进行二次精炼（可关闭）

## 目录结构
- `animetrace_plugin/plugin.py`：Tool 实现与插件注册、图片解析与输出格式化、配置 Schema
- `animetrace_plugin/trace_client.py`：AnimeTrace API 客户端（httpx、重试/退避、缓存、响应格式化）
- `animetrace_plugin/_manifest.json`：插件清单（components=tool: anime_trace_search）
- `animetrace_plugin/animetrace_demo/`：接口调用与示例数据

## 安装与启用
1) 将 `animetrace_plugin/` 放入 MaiBot 插件目录（通常为 `plugins/`）
2) 重启 MaiBot，系统会自动为插件生成 `config.toml`
3) 在生成的配置中启用插件（`[plugin] enabled = true`）

注意：不要手动创建 `config.toml`，请以 `plugin.py` 中的 `config_schema` 为准（系统自动生成）

## 依赖
- Python 依赖：`httpx>=0.24.0`
  - 安装：`pip install httpx`
  - 未安装时调用会提示缺失

## 配置项
`config.toml` 的注释说明来源于 `config_schema`，以下为关键项与默认值：

```
[plugin]
config_version = "2.2.0"   # 配置文件版本
enabled = true             # 是否启用插件

[anime_trace]
endpoint = "https://api.animetrace.com/v1/search"  # AnimeTrace 接口地址
model = "animetrace_high_beta"                      # 默认识别模型
is_multi = 1                                        # 是否返回多条候选（0/1）
ai_detect = 1                                       # 是否开启AI图检测（1=开启，2=关闭）
request_timeout = 15.0                              # HTTP 请求超时（秒）
min_similarity = 0.0                                # 最小相似度阈值（百分比 0–100）
history_lookup_seconds = 3600.0                     # 历史消息回溯窗口（秒）
history_lookup_limit = 20                           # 最多扫描的历史消息条数
use_image_selector_llm = true                       # 是否使用 LLM 选择历史消息中的图片
image_selector_model_key = "plugin_reply"           # 用于图片选择的模型键名
use_llm = false                                     # 是否使用 LLM 总结识别结果（暂时无效）
llm_model_key = "plugin_reply"                      # 用于最终结果摘要的模型键名（暂时无效）
```

## 工具参数（anime_trace_search）
- `use_reply`（bool）是否优先使用被回复的消息中的图片（默认 false）
- `image_index`（int）选中的图片序号（从 1 开始）；支持负数（-1 表示最后一张）
- `is_multi`（int）AnimeTrace 是否返回多条候选（0/1），未传入时使用配置默认值
- `model`（string）AnimeTrace 模型名，未传入时使用配置默认值
- `ai_detect`（int）是否开启 AI 图检测（1=开启，2=关闭），未传入时使用配置默认值
- `min_similarity`（string）最小相似度阈值，支持小数或百分比字符串，如 `"0.87"` 或 `"87%"`

返回结果为文本摘要，包含：
- 一行主要结论（角色 TopN 或最优候选与相似度）
- 最多 5 条候选行（受相似度阈值过滤）
- 服务返回的 `trace_id`（如果存在）
- 开启 `use_llm` 时，可能被 LLM 精炼为更短的中文摘要（暂时无效）

## 工作流程概览
1) 图片选择：优先回复消息 → 当前消息 → 历史消息（可启用 LLM 选择器）
2) 编码与上传：支持 URL / Base64 / CQ 码 / picid；URL 下载带重试与缓存
3) AnimeTrace 调用：JSON 请求，httpx 连接池 + 429/5xx 退避重试
4) 结果格式化：统一相似度为百分比，角色/作品优先级展示
5) 可选总结：配置 `use_llm=true` 时由 LLM 生成 3 行内结论（暂时无效）

## 使用示例
- 由 LLM 自动选择使用该工具（常规对话中带图请求“搜番/识别出处”）
- 在插件代码中直接调用：

```python
tool = AnimeTraceTool(plugin_config, chat_stream)
result = await tool.direct_execute(use_reply=True, image_index=-1)
print(result["content"])  # 文本摘要
```

## 日志与可观测性
- 请求体日志自动脱敏（Base64 替换为长度与前缀信息）
- 记录请求/下载的状态码与内容长度，便于排障
- 建议在生产环境将候选详细列表降为 debug 级别

## 故障排查（FAQ）
- HTTP 400/参数错误：尝试 `ai_detect=2`、`is_multi=0`、更换 `model`（如 `pre_stable`）
- 413 图片过大：改为 URL 上传或缩小图片尺寸
- 503/429：服务繁忙/限流，已自动退避；如仍失败请稍后重试
- 未找到图片：请确认消息中包含 URL/Base64/CQ 码或图片被数据库记录（picid）
- URL 下载失败：确保 URL 可公网直连；或改为提供 Base64 内容

## 感谢
- [晴空](https://github.com/XXXxx7258)