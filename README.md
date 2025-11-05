# AnimeTrace 图片识别插件（MaiBot）

一个用于接收 QQ 图片并调用 AnimeTrace 识别服务的插件。识别完成后，插件会将提取到的关键信息（相似度/标题或角色TopN摘要）以文本形式回复到原会话。

## 功能特性
- 支持三种入参形式：`url`、`base64`、`file`（自动选择与兜底）。
- 自动监听图片消息或通过指令触发（可配置）。
- 与 Napcat 适配：递归解析回复/转发/seglist 等嵌套消息中的图片字段（含 `data.file`）。
- 友好的 info 级日志：包含请求体摘要（base64 脱敏与截断）、响应状态码、解析摘要。
- 可配置的模型/相似度阈值/是否多结果/AI图检测等参数；高扩展、低耦合。

## 目录结构
- `animetrace_plugin/plugin.py`：事件处理器与插件注册，消息解析与发送回执。
- `animetrace_plugin/trace_client.py`：AnimeTrace API 客户端（httpx / JSON 提交）。
- `animetrace_plugin/_manifest.json`：插件清单（声明事件处理器组件）。
- `animetrace_plugin/config.toml`：插件默认配置。

## 安装与依赖
- Python 依赖：建议安装 `httpx`（用于 HTTP 请求与文件上传）。
  - `pip install httpx`
- 插件被加载后，如未安装 httpx，会在调用时提示缺失。

## 启用方式
1. 确保宿主已扫描到 `animetrace_plugin/` 目录（PluginManager 插件目录）。
2. 在配置中启用插件：
   - `animetrace_plugin/config.toml` 中 `[plugin] enabled = true`
3. 启动 MaiBot；控制台应出现组件注册日志。

## 配置项（config.toml）
```toml
[plugin]
config_version = "1.0.0"
enabled = true

[anime_trace]
# AnimeTrace API
endpoint = "https://api.animetrace.com/v1/search"
model = "animetrace_high_beta"   # 可选 pre_stable 等
is_multi = 1                      # 0/1：是否返回多条
ai_detect = 1                     # 1：开启；2：关闭
request_timeout = 15.0

# 触发策略
auto_on_image = true              # 收到图片自动识别
trigger_keywords = ["/trace", "动漫识别", "以图搜番"]
max_images = 1
min_similarity = 0.0              # 支持 0–1 或 0–100
input_preference = "prefer_base64"  # auto|prefer_url|prefer_base64
```
说明：
- `input_preference` 决定 URL 图片走 `url` 直传还是转 `base64` 上传；`prefer_base64` 会自动下载并转码；失败时会回退为文件上传（由插件处理，不需额外配置）。
- `min_similarity` 用于过滤结果；支持字符串百分比的解析（如 "87.5%"）。

## 触发方式
- 自动：当 `auto_on_image = true` 且消息中包含图片时自动触发。
- 指令：当 `auto_on_image = false` 时，通过关键字触发（默认 `/trace`），支持“回复图片 + /trace”。

## 日志说明（info）
- 请求摘要（已脱敏）：
  - `[AnimeTrace] Request body(JSON default): {'model': '...', 'is_multi': 1, 'ai_detect': 1, 'base64': '<base64 len=... head=...>'}`
- 请求状态：
  - `[AnimeTrace] POST ... status=200 content-length=...`
- 图片输入侧：
  - `Image #1 url=...` 或 `Image #1 base64 len=... head=...` 或 `url-to-base64 len=...`
- 响应摘要：
  - `Resp keys=[...] data_type=list data_len=...`

## 故障排查（FAQ）
- 400 参数错误：
  - 切换 `ai_detect` 为 `2`；切换 `model` 为 `pre_stable`；切换 `is_multi` 为 `0`；尝试 `input_preference = "prefer_url"`。
  - 若仍然 400，请附上两行日志（首次 POST 的 `status` 与错误摘要 `code/msg`）。
- “未指定图片”：
  - 确认消息里至少包含其一：`url`/`base64`/`file`；对“回复图片 + 指令”的情况，已支持从嵌套结构提图。
- Base64 过长：
  - 使用文件上传（`file`）或 URL 直传；命令行 cURL 可用 `--data-binary @payload.json` 避免参数过长。
- 无法下载 URL：
  - 确保 URL 可公网直连（非临时/权限受限）；或改为 `prefer_base64`。

## 二次开发
- 事件处理器：`AnimeTraceOnMessage`（`EventType.ON_MESSAGE`）。
- API 客户端：`AnimeTraceClient.search(url|base64|file, model, is_multi, ai_detect)`
- 解析逻辑：
  - 优先展示角色类结果（`data[0].character`）为 `角色TopN: work:character; ...`
  - 其次展示相似度类结果（`similarity/score/sim` + `title/name/source`）。
- 日志脱敏：`trace_client._build_log_payload()` 统一处理，禁止输出完整 Base64。

## 变更与兼容
- v1.0.0：
  - 初版：JSON 提交为主；已去除表单回退与 urllib 回退；Base64→文件上传兜底保留。
  - 增强：Napcat 嵌套消息解析；角色摘要；相似度百分比解析；请求/响应日志。

---

若需要我为你的运行环境补充示例配置或做特定服务端字段兼容（如定制 `ai_detect/is_multi` 取值范围），请告知实际返回的 `code/message`。
