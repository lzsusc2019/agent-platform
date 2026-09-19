# Agent Platform (灵枢智能体中台 · MVP)

> 可恢复、可中断、可审批的智能体中台骨架。

把 `Agent中台.md` 的设计落到代码的第一版。范围刻意收紧：先跑通
**Agent Loop + Checkpoint + HITL** 这条主路径，沙箱 / 热加载 / MCP 留接口和 TODO。

附带一个 Admin Dashboard，在 `/admin/`，单页：配置查看编辑、Checkpoint 检视、
HITL 审批、调试 chat。

---

## 1. 快速开始

```bash
# 安装（uv 可用则 uv sync --extra test）
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"

# 跑测试：不需要 Redis，不需要网络，用 fakeredis + mock 模型
.venv/bin/python -m pytest

# 启动服务
.venv/bin/python -m agent_platform.cli serve        # 本机 Redis 已在跑
AGENT_PLATFORM_USE_FAKE_REDIS=true \
  .venv/bin/python -m agent_platform.cli serve       # 没有 Redis 时
```

然后打开 <http://localhost:8000/admin/>。

**没配 DeepSeek key 也能跑**：平台会 fallback 到 mock 模型并在日志里打 WARNING，
界面和 HITL 全流程照常可调试，只是模型回的是 mock 文本。

---

## 2. 范围

### 包含

| 能力 | 说明 |
|---|---|
| **Agent Loop** | ReAct 推理循环、上下文阈值压缩、指数退避重试、空响应兜底、100 轮硬熔断 |
| **Checkpoint** | Redis 持久化运行时状态（messages、pending_tools、done_results），TTL 可配 |
| **HITL** | 敏感工具 → 挂起 → SSE 通知 → 审批 → 从 Checkpoint 恢复继续 |
| **审批授权** | 落盘、限范围、有时效、可撤销、可审计 |
| **工具策略** | 按 agent 裁剪可用工具；敏感集只能加不能减 |
| **SSE 流式** | 9 种事件：`start` / `assistant` / `tool_call` / `tool_result` / `compressed` / `hitl_required` / `hitl_resolved` / `finish` / `error` |
| **工具** | `echo`、`http_get`，以及标记为敏感、需审批的 `write_file` |
| **Admin Dashboard** | 6 个 tab，见第 4 节 |

### 明确不做

- 鉴权 / 租户隔离 —— Dashboard 无任何认证，**只能本地或内网使用**
- 沙箱 / 双容器 / 凭证代理
- Agent 实例热加载（当前是进程内 Map + 变更失效，没有轮询线程）
- MCP 客户端
- MySQL / 对象存储；消息与会话的持久化只在 Redis 里
- 灰度 / 双机房

---

## 3. 架构与代码分层

```
┌──────────────┐  SSE   ┌────────────────────┐
│   Frontend   │<------>│  FastAPI  (api/)   │
└──────────────┘        └─────────┬──────────┘
                                  │
                        ┌─────────▼───────────────┐
                        │   Agent Engine          │
                        │ (domain/agent_loop.py)  │
                        │                         │
                        │  ┌───────────────────┐  │
                        │  │ before_model      │  │
                        │  │  · 上下文压缩     │  │
                        │  │  · token 估算     │  │
                        │  ├───────────────────┤  │
                        │  │ llm_call          │  │
                        │  │  · 指数退避重试   │  │
                        │  │  · 空响应兜底     │  │
                        │  ├───────────────────┤  │
                        │  │ after_model       │  │
                        │  │  · 工具白名单     │  │
                        │  │  · 审批授权检查   │  │
                        │  │  · HITL 挂起      │  │
                        │  ├───────────────────┤  │
                        │  │ tool_executor     │  │
                        │  │  · 幂等键         │  │
                        │  │  · 越权拒绝       │  │
                        │  └───────────────────┘  │
                        └──┬───────────┬──────────┘
                           │           │
                   ┌───────▼───┐  ┌────▼─────────┐
                   │ Checkpoint│  │ Tool         │
                   │ (Redis)   │  │ Registry     │
                   └───────────┘  └──────────────┘
```

```
agent-platform/
├── src/agent_platform/
│   ├── config/      配置：Settings 四层解析 + agent 种子
│   ├── domain/      领域：agent 本身，不碰 IO
│   │                  messages / events / errors / checkpoint / tool
│   │                  llm（ChatModel 端口 + Mock）/ agent_loop
│   ├── infra/       基础设施：所有对外 IO
│   │                  checkpoint_store / config_store / secrets_store
│   │                  approval_store / providers（DeepSeek）/ agent_manager
│   ├── api/         HTTP：app / routes（/v1）/ admin（Dashboard）
│   └── tools/       工具实现：echo / http_get / write_file
├── resource/        归档资源，不在包内
│   ├── static/        admin.html（Dashboard 页面）
│   └── tests/         测试套件
├── config/          配置归档（platform.yaml 本机私有，见第 5 节）
├── docs/adr/        决策记录
└── scripts/         验证脚本
```

**依赖方向单向**：

```
config ──┐
         ├──> domain <── infra <── api
tools ───┘
```

- `domain/` **不 import** `infra/` 或 `api/`。这是它可测的根本原因：ReAct 循环能在
  没有 Redis、没有 HTTP 的情况下跑完整套逻辑。
- `infra/` 实现 domain 定义的端口（`ChatModel`、`Tool`），依赖指向 domain。
- `api/` 只做协议转换，不含业务规则。

> `resource/` 在**仓库根**，代价是进不了 wheel：`DASHBOARD_PATH` 靠 `project_root()`
> 向上找 `pyproject.toml` 解析，装到 site-packages 后没有仓库根，`/admin/` 会返回 503
> 并说明原因。对一个内部调试面板，这是划算的取舍。

### 决策记录（ADR）

架构上「为什么不那样做」都写在 ADR 里，读代码前先看这几篇能省很多「这写得不对吧」：

| ADR | 主题 |
|---|---|
| [ADR-001](docs/adr/ADR-001-mvp-scope.md) | MVP 范围与边界 —— 什么刻意不做 |
| [ADR-002](docs/adr/ADR-002-loop-implementation.md) | 为什么 ReAct Loop 自己实现，而不是交给 LangGraph |
| [ADR-003](docs/adr/ADR-003-checkpoint.md) | Checkpoint 存储模型与幂等键（PENDING / DONE / FAILED 三态） |
| [ADR-004](docs/adr/ADR-004-hitl.md) | HITL 三状态机、审批授权语义、恢复时的消息顺序不变式 |
| [ADR-005](docs/adr/ADR-005-configuration.md) | 配置分层，代码里不留字面量 |

---

## 4. Admin Dashboard

访问 <http://localhost:8000/admin/>，6 个 tab：

| Tab | 作用 |
|---|---|
| **Agents** | 列出所有 `agent_id` 与编辑表单（system_prompt / model / temperature / max_tokens / tools / sensitive_tools / skills）。保存后 `AgentManager` 失效缓存，下一次 chat 重建 AgentLoop |
| **Debug Chat** | 直接发 SSE chat：agent_id、content、可选 thread_id（空则自动生成）、approval_id。触发 HITL 后 thread_id 和 approval_id **自动回填**，点 **Approve & Resume** 直接闭环 |
| **Checkpoints** | 所有 thread_id、状态徽章（running / waiting_approval / finished），点开看完整 snapshot JSON |
| **HITL Pending** | 等待审批的 thread + 触发工具名 + 每行 approve / reject（批准后自动跳到 Debug Chat 并立即恢复）。下方 **Active approvals** 列出生效中的授权（工具 / 目标 / 剩余时效），可逐条 revoke |
| **Tools** | 当前注册的所有工具，敏感工具标红 |
| **Providers** | 配 DeepSeek key。保存即生效，**无需重启**；带 test 按钮直接打 `/models` 验连通性 |

> ⚠️ **Dashboard 没有任何鉴权**，`/admin/api/*` 可直接读写配置和密钥。只能本地或内网用，
> 不要暴露到公网。

### 配 DeepSeek key（不用重启）

二选一：

1. **页面**：Providers tab → deepseek / api_key 行点 set → 保存 → 点 test 验连通性
2. **文件/环境变量**：见第 5 节「密钥」，启动日志会打印来源

然后切到 **Debug Chat** 选 `demo` 发条消息。`start` 事件里的 `llm` 字段会显示
`DeepSeekChatModel` —— 如果显示 `MockChatModel`，说明 key 没生效。

Key 永远只以 `sk-4********0f51` 这种遮罩形式显示。

### Admin API

```bash
# 列出所有 agent 配置
curl http://localhost:8000/admin/api/configs

# 写/改一个 agent（tools 留空 = 不限制，见第 6 节）
curl -X PUT http://localhost:8000/admin/api/configs/qa \
  -H "content-type: application/json" \
  -d @- <<JSON
{"system_prompt":"you are a QA bot","tools":["echo"],"sensitive_tools":[]}
JSON

# 调试 chat（SSE）
curl -N -X POST http://localhost:8000/admin/api/chat \
  -H "content-type: application/json" \
  -d @- <<JSON
{"agent_id":"qa","content":"please echo hello"}
JSON

# 待审批 / 生效中的授权 / checkpoint
curl http://localhost:8000/admin/api/hitl/pending
curl http://localhost:8000/admin/api/approvals
curl http://localhost:8000/admin/api/checkpoints/<thread_id>
```

---

## 5. 配置

### 四层解析

平台**不硬编码任何可调参数**，优先级由低到高：

```
1. 代码默认值                  src/agent_platform/config/settings.py
2. config/platform.yaml        本机平台配置（不进版本控制，见下）
3. config/platform.local.yaml  可选每机覆盖
4. 环境变量 / .env             AGENT_PLATFORM_*
```

另加 `Settings(**kwargs)` 显式传参压过全部（测试用这层）。

```bash
# 打印当前生效值（密钥自动脱敏）
.venv/bin/python -m agent_platform.cli config
```

**所有配置入口都能容忍工作目录不对**：`resolve_config_path()` 先试 CWD，找不到就回退到
项目根（向上找 `pyproject.toml`）。这条不是可有可无 —— 填错工作目录的后果不是报错，
而是**静默降级成 MockChatModel**，见第 9 节。

### 密钥

**`config/platform.yaml` 不进版本控制，仓库是公开的。** 它载着你的 key，一旦提交就
永久留在 git 历史里，`git log -p` 随时能翻出来 —— 事后擦不干净。

所以本机要自己放一份：

```yaml
# config/platform.yaml   （.gitignore 掉）
deepseek_api_key: sk-xxxxxxxx
```

或者用环境变量（优先级更高，适合 CI / 容器）：

```bash
export AGENT_PLATFORM_DEEPSEEK_API_KEY=sk-xxxxxxxx
```

**接受的写法**（都是同一个值）：

| 来源 | 写法 |
|---|---|
| YAML / kwargs | `deepseek_api_key` 或别名 `deepseek_apikey` |
| 环境变量 | `AGENT_PLATFORM_DEEPSEEK_API_KEY` / `AGENT_PLATFORM_DEEPSEEK_APIKEY` |
| 环境变量回退 | `DEEPSEEK_API_KEY`（官方 SDK 的变量名） |

启动日志会打印来源与打码后的值：

```
INFO config.llm provider=DeepSeekChatModel model=deepseek-flash \
     key=sk-46fc...0f51 source=config/platform.yaml
```

- key 前后空格自动 trim；含非 ASCII 或换行会在**保存时**被 400 拦下，不用等发请求
- 测试通过 `AGENT_PLATFORM_PLATFORM_YAML_FILE` 指向临时空文件，**不会读你本地的 key**

### config/platform.yaml

可 diff、可 review 的配置正本。**只 pin 你确实要偏离代码默认值的项** —— 省略的键会继续
跟随代码默认值演进。

```yaml
max_turns: 50              # 这个环境任务短，收紧到 50
deepseek_model: deepseek-flash
tool_http_get_timeout: 3.0 # 内网快，超时压短
```

> `resource/tests/test_configurability.py` 会断言 YAML 里 pin 的值与代码默认值一致。
> 该文件不在版本控制里，所以**这个保证只在有该文件的机器上有效**；文件不存在时测试
> 会 skip 而不是失败。

### config/agents.yaml

agent 定义（工具列表、多行提示词、嵌套 metadata）塞进环境变量既难写又难 review，
所以单独归档成 YAML，**这个文件进版本控制**（不含密钥）：

```yaml
agents:
  - agent_id: demo
    model: deepseek-flash
    system_prompt: |
      You are a helpful assistant. Use the available tools when the user
      asks you to.
    temperature: 0.7
    max_tokens: 2048
    tools: [echo, http_get, write_file]
    sensitive_tools: [write_file]
```

- 启动时按 `agent_id` 写入运行时存储，**已存在的跳过** → Dashboard 改过的不会被重启冲掉
  （代价见第 9 节「陈旧的 agent」）
- 省略的字段回落到 `agent_default_*` 设置，最简可以只写 `agent_id` + `model`
- YAML 语法错 / 字段类型错 → 抛 `AgentSeedError` **拒绝启动**（带着半个配置静默起来更糟）
- 换路径或关闭种子：`AGENT_PLATFORM_AGENT_SEED_FILE=/path/to/agents.yaml`，设为空串即关闭

### 环境变量参考

前缀统一 `AGENT_PLATFORM_`。只列常用的，完整列表见 `config/settings.py`。

**存储**

| 变量 | 默认 | 说明 |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 连接串 |
| `USE_FAKE_REDIS` | `false` | 用内存 fakeredis，重启即丢，**不要用于生产** |
| `CHECKPOINT_TTL_SECONDS` | `1800` | 挂起的 thread 能等多久 |
| `APPROVAL_GRANT_TTL_SECONDS` | `3600` | 一次审批签发后能用多久（与上一条互相独立） |

**Agent Loop**

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_TURNS` | `100` | 单轮硬熔断 |
| `LLM_MAX_RETRIES` | `3` | LLM 重试次数 |
| `LLM_RETRY_BASE_DELAY` | `0.5` | 指数退避基数（秒） |
| `EMPTY_RESPONSE_MAX_RETRIES` | `3` | 空响应兜底重试 |
| `COMPRESS_TRIGGER_TOKENS` | `180000` | 超过即触发上下文压缩 |
| `COMPRESS_KEEP_RECENT_TURNS` | `30` | 压缩时保留最近多少条 |
| `TOKEN_ESTIMATE_CHARS_PER_TOKEN` | `4` | 粗略 token 估算比例 |

**Agent 默认值**（新建 agent 且字段缺省时使用）

| 变量 | 默认 |
|---|---|
| `AGENT_DEFAULT_SYSTEM_PROMPT` | `You are a helpful assistant. Use tools when needed.` |
| `AGENT_DEFAULT_MODEL` | `mock` |
| `AGENT_DEFAULT_TEMPERATURE` | `0.7` |
| `AGENT_DEFAULT_MAX_TOKENS` | `4096` |
| `AGENT_SEED_FILE` | `config/agents.yaml` |

**DeepSeek**

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 空 | 见上文「密钥」 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | |
| `DEEPSEEK_MODEL` | `deepseek-flash` | `model: deepseek` 时用它；也可写 `deepseek:deepseek-v4-pro` |
| `DEEPSEEK_TIMEOUT` | `30.0` | |
| `DEEPSEEK_REQUIRE_KEY` | `false` | 设 true 则缺 key 时启动直接抛错，而不是降级 mock |

**工具**

| 变量 | 默认 | 说明 |
|---|---|---|
| `TOOL_HTTP_GET_TIMEOUT` | `5.0` | |
| `TOOL_HTTP_GET_MAX_BODY_CHARS` | `500` | 回灌给模型的响应体截断长度 |
| `TOOL_HTTP_GET_ALLOWED_SCHEMES` | `["http://","https://"]` | 协议白名单 |
| `TOOL_WRITE_FILE_ROOT` | `workspace` | `write_file` 可写根目录，越界路径直接拒绝 |
| `TOOL_WRITE_FILE_MAX_BYTES` | `65536` | 单次写入字节上限（按 UTF-8 字节算） |

**Admin / 观测**

| 变量 | 默认 | 说明 |
|---|---|---|
| `ADMIN_SECRET_TEST_TIMEOUT` | `10.0` | Provider key 连通性检测超时 |
| `LOG_LEVEL` | `INFO` | |

---

## 6. 工具策略与 HITL

### 工具策略（按 agent 裁剪）

`AgentConfig` 里两个字段决定「这个 agent 能用什么、哪些要审批」：

| 字段 | 语义 |
|---|---|
| `tools` | 可用工具白名单。**留空 = 不限制**（全部注册工具），不是「禁用所有工具」 |
| `sensitive_tools` | 额外需要审批的工具。**只能加，不能减** |

两条都是刻意的选择：

- **空 `tools` = 不限制**：所有既有配置和 Dashboard 新建的 agent 这个字段都是空的，
  解释成「没有工具」会把它们全部悄悄缴械。
- **`sensitive_tools` 只能加不能减**：代码里标了 `sensitive = True` 的工具（`write_file`）
  永远需要审批，配置漏写或写空都不解除。审批闸门不该被一个配置笔误关掉。想给本来
  安全的工具加审批（比如让 `http_get` 也要批）才用它。

白名单是**执行期强制**的，不只是「不给 schema」。schema 列表只是告知；被幻觉或被提示
注入的模型仍然可以点名一个没见过的工具，所以 Loop 在真正执行前会再查一次白名单，
越权调用以 `is_error` 结果回灌给模型（`AgentLoop._execute_tools`）。

推论：**不被允许的工具不会触发审批弹窗**。既然答案无论如何都是「不」，问人只是噪音。

### 一次审批到底批准了什么

「已批准」不是请求里的一个字段，而是 **Redis 里的一条记录**。`/hitl/approve` 写入它，
恢复时 Loop 查它 —— **不看请求里带了什么**。

| 属性 | 语义 |
|---|---|
| **范围** | 一次审批覆盖**一个目标**。`write_file` 的范围是解析后的绝对路径，所以批准 `notes/a.txt` **不**授权 `notes/b.txt` |
| **时效** | `approval_grant_ttl_seconds`，默认 1 小时。过期后同一个文件重新弹审批 |
| **主体** | 签发给 `(agent_id, user_id)`，别人的回合不继承 |
| **可撤销** | Dashboard 的 Active approvals 面板能看能撤 |

实际效果：

| 操作 | 结果 |
|---|---|
| 批准 `notes/a.txt`，一小时内再写 `notes/a.txt` | 免审批直接执行（哪怕换了会话） |
| 写 `notes/b.txt` | **重新审批** |
| 换个 `user_id` 写 `notes/a.txt` | **重新审批** |
| 一小时后写 `notes/a.txt` | **重新审批** |
| 跳过 `/hitl/approve` 直接 resume | **重新挂起**（`reason=grant_missing_or_expired`），工具不执行 |
| 伪造 `appr_xxx_随便编` | **400** |

> `notes/a.txt`、`./notes/a.txt`、`notes/../notes/a.txt` 解析成同一路径，共享同一次授权；
> 而 `../escape.txt` 解析失败，永远不会匹配到别人合法写入挣来的授权。

> 早先的实现里 `/hitl/approve` **什么都不写**，Loop 只判断「请求里有没有 approval_id」。
> 后果是：跳过 approve 直接 resume 也能跑通、只校验前缀所以伪造 id 也能过、而且**一次
> 批准放行整个回合**的所有敏感调用。现在权威在服务端记录里。

### 怎么触发并验证 HITL

让模型去调 `write_file` 就行 —— 它在代码里标了 `sensitive`：

```bash
SID=$(curl -s --noproxy "*" -X POST http://127.0.0.1:8000/v1/sessions \
  -H "content-type: application/json" \
  -d @- <<JSON | sed "s/.*\"thread_id\":\"\([^\"]*\)\".*/\1/"
{"agent_id":"demo","user_id":"u1"}
JSON
)

curl -N --noproxy "*" -X POST http://127.0.0.1:8000/v1/sessions/$SID/chat \
  -H "content-type: application/json" \
  -d @- <<JSON
{"content":"Use the write_file tool to save hello into notes/hitl.txt"}
JSON
# -> event: hitl_required   "approval_id": "appr_<thread>_<call>"
```

> **措辞很关键**：要明确点名工具和路径。含糊的 "write a file" 真模型经常直接回一段话
> 而不调工具。

一键全链路（含 approve 和 reject 两条分支，以及真正重要的两条断言 —— 批准前工具
绝不能执行、驳回后绝不能落盘）：

```bash
scripts/hitl_walkthrough.sh          # 真 API，服务需在跑
.venv/bin/python scripts/smoke.py    # 离线 mock 版，秒级
```

或者在 Dashboard 里点：**Debug Chat** 发一条写文件的请求 → 看到 `hitl_required`
（thread_id 和 approval_id 自动回填）→ 点 **Approve & Resume**。也可以在 **HITL Pending**
tab 里对任意一条待审批直接点 approve / reject。

---

## 7. LLM 供应商配置

`AgentConfig.model` 决定走哪个供应商，格式 `provider[:model_name]`：

| `model` 字符串 | 行为 |
|---|---|
| `mock` | `MockChatModel`，确定性、无网络，测试与缺 key 时的兜底 |
| `deepseek` | DeepSeek，用 `settings.deepseek_model`（默认 `deepseek-flash`） |
| `deepseek:deepseek-v4-pro` | DeepSeek，指定 Pro 模型 |
| `openai:gpt-4o-mini` | 报错（未注册）；先 `register_provider("openai", ...)` 即可 |

**没设 key 会怎样**：默认 fallback 到 `MockChatModel` 并打 `WARNING provider.fallback`。
界面照常可调试，只是模型回的是 mock 文本。想让它立刻失败（生产建议）：

```bash
export AGENT_PLATFORM_DEEPSEEK_REQUIRE_KEY=true   # 启动时抛 RuntimeError
```

### 加新供应商（OpenAI、Anthropic、本地模型…）

实现 `ChatModel` 接口（看 `domain/llm.py`）然后注册：

```python
from agent_platform.infra.providers import register_provider
from agent_platform.domain.llm import ChatModel, LLMResponse

class OpenAIChatModel(ChatModel):
    async def ainvoke(self, messages, tools):
        ...
        return LLMResponse(content=..., tool_calls=[...])

register_provider("openai", OpenAIChatModel)
```

之后 `model: openai:gpt-4o-mini` 就会路由到这里。参考 `DeepSeekChatModel`
（`infra/providers.py`）—— 整段约 80 行，因为 DeepSeek 与 OpenAI 的 API 兼容，
直接 copy 改 base_url 和鉴权 header 即可。

> 这是 ADR-002 明确否决 LangChain / LangGraph 的原因：接入一个供应商需要的是
> 一个 httpx 子类，不是一个框架。项目曾经把 `langchain-core` 和 `langgraph` 挂在
> 依赖里却从未 import，后来删掉了。

---

## 8. 验证脚本

按「要不要花真钱」排序：

| 脚本 | 依赖 | 用途 |
|---|---|---|
| `scripts/smoke.py` | 无外部服务 | 端到端跑一遍 HTTP 面：healthz → 建会话 → SSE chat → 工具回合 → HITL 挂起 → 审批 → 恢复 → admin 接口。**agent 被显式 pin 到 mock**，不联网、不花钱、断言确定 |
| `scripts/diag_orderings.py` | 无外部服务 | 复现 Dashboard 上 5 种操作顺序，排查「先点了 A 再点 B」类问题 |
| `scripts/probe_natural_language.py` | 真 API | 12 种自然语言说法 × 2 种 system prompt，实测哪些能触发工具调用 |
| `scripts/hitl_walkthrough.sh` | 真 API + 服务在跑 | HITL 全链路，含 reject 分支 |
| `scripts/live_verify.py` | 真 API | 工具回合、`http_get` 真实取数、敏感工具 HITL，并断言出站消息顺序合法 |
| `scripts/check_approval_scope.py` | 真 API + 服务在跑 | 审批范围：同文件免审 / 换文件必审 / 换用户必审 / 跳过审批不执行 / 伪造 id 被拒 / 撤销后恢复审批 |

```bash
.venv/bin/python scripts/smoke.py
.venv/bin/python scripts/diag_orderings.py
env -u http_proxy -u https_proxy .venv/bin/python scripts/live_verify.py
```

> 本机若设了 `http_proxy`，跑 curl 或 live 脚本要加 `--noproxy "*"` /
> `env -u http_proxy`，否则请求会绕到代理上。

---

## 9. 调试与踩过的坑

这一节记的都是**实际踩过、且症状会误导人**的东西。每一条都配了它为什么难查。

### 在 PyCharm / VS Code 里断点调试

新建 Python Run/Debug Configuration：

| 字段 | 值 |
|---|---|
| Python interpreter | `<项目根>/.venv/bin/python` |
| **Module name**（不是 Script path） | `agent_platform.cli` |
| Parameters | `serve --host 127.0.0.1 --port 8000` |
| Working directory | `<项目根>`（含 `pyproject.toml` 的那层） |

入口是 **module 而不是 script** —— `cli.py` 直接跑会被子命令解析吃掉（`main()` 只特判
`check`）。跑测试再建一个：Module name `pytest`，Parameters `-q`。

#### 不要勾 `--reload`

uvicorn 的 reloader 会 **fork 一个子进程**跑真正的 app，而调试器挂在父进程上 ——
结果是服务正常跑、**断点一个都不命中**，而且不报任何错。人会以为是断点位置不对，
白排查半小时。要热加载就用普通 Run，要调试就用 Debug，别混。

#### 值得下断点的位置

| 位置 | 看什么 |
|---|---|
| `domain/agent_loop.py` → `run()` | 整个 ReAct 循环、恢复分支、100 轮熔断 |
| `domain/agent_loop.py` → 审批检测处 | 哪一次 tool call 触发了 HITL |
| `infra/providers.py` → `DeepSeekChatModel.ainvoke()` | **真实出站 payload**，排查 400 / 422 必看 |
| `tools/builtins.py` → `WriteFileTool.run()` | 审批放行后的实际写盘，断在这里能确认路径校验 |
| `infra/agent_manager.py` → `_resolve_llm()` | 这次到底解析成了哪个 ChatModel |

先加上 `AGENT_PLATFORM_LOG_LEVEL=DEBUG`。

### 消息顺序不变式（这个坑踩了四次）

OpenAI 兼容接口有一条硬规则：

> **带 `tool_calls` 的 assistant 消息，后面必须紧跟应答它每一个 `tool_call_id` 的
> tool 消息，中间不能出现别的角色。**

违反它是一句 400，只说「第几条消息」，所以看起来像线格式 bug，真实原因却往往是
**别的路径把历史弄脏了**。四次踩法：

| # | 踩法 | 症状 | 修法 |
|---|---|---|---|
| 1 | 序列化格式错：`tool_calls` 少了 `type`、`arguments` 没编码成 JSON 字符串 | `422 missing field type` | `Message.to_openai_dict()` |
| 2 | 恢复请求带 `content: ""`，空串不是 `None`，被当成真实用户消息插进中间 | `400 insufficient tool messages` | 入口把空白 `user_message` 归一成 `None`；用户消息**延后追加** |
| 3 | 上下文压缩按条数切分，正好切在 assistant 和它的 tool 结果中间 | 同上（长会话） | 切分后把队首孤儿 tool 消息推进 `older` |
| 4 | **驳回审批后那条调用永远没人应答** | 驳回当次没事，**之后**往同一 thread 发消息就 400 | `/hitl/reject` 补一条如实说明的回执 |

**第 4 条是唯一跑到线上的**，它揭示了一个模式：只要每个调用方各自维护这个不变式，
就总会有人漏。所以现在 Loop 在**每次调用模型之前**重建消息序列
（`repair_tool_call_ordering`）：缺的回执补在紧跟 assistant 的位置、孤儿 tool 消息丢掉，
并打 `loop.repaired_message_history` WARNING。

两个细节值得记：

- **必须「插入」而不是「追加」**。追加看着能修，其实不行 —— provider 要求回执**紧邻**
  调用；如果中间已经插进了用户消息，追加到末尾等于没修。这是兜底代码第一版的错误，
  被测试抓出来了。
- 打了 WARNING 意味着**别的层出了问题**，值得去看，别当成正常路径。

回归测试：`resource/tests/test_wire_format.py`、`test_hitl_resume_wire.py`、
`test_tool_call_ordering.py`。后者用一个**故意越权的桩模型**模拟幻觉 / 提示注入 ——
`MockChatModel` 太规矩，只调被提供的工具，测不出这类洞。

### 路径解析一律不数 `..`

**这个坑踩了两次，症状都是「配置文件不见了」。**

1. `project_root()` 用 `parents[2]` 找仓库根。`config.py` 下沉一层变成
   `config/settings.py` 后，它静默指向了 `src/agent_platform`。
2. 测试用 `Path(__file__).parent.parent` 找仓库根。`tests/` 移进 `resource/` 后，
   **5 个用例全挂**，报的是「config/platform.yaml is missing from the repo」，
   看着像仓库坏了，其实是路径过期。

现在两边都**向上找 `pyproject.toml`**（`project_root()`），测试通过 `repo_root` fixture
复用它。测试和应用对「仓库根在哪」的定义由构造保证一致，再移文件也不会走偏。

更早的一个变体是**工作目录**：所有配置入口都是相对路径，填错 CWD 的后果不是报错，
而是**静默降级成 MockChatModel** —— agent 照常回话，只是回 mock 文本，日志里只有一行
WARNING。`resolve_config_path()` 现在会在 CWD 找不到时回退到项目根。

### Redis 里的陈旧 agent

`seed_defaults()` 只在 `agent_id` **不存在**时写入。这样设计是为了不冲掉 Dashboard 上
的改动，代价是：**你改了 `config/agents.yaml` 之后重启，已存在的 agent 不会被更新**。

典型症状是改了 seed 却毫无反应，chat 一直回 mock 的 `ok`。清掉重来：

```bash
curl -X DELETE http://127.0.0.1:8000/admin/api/configs/demo   # 或 Dashboard 里删
# 然后重启服务，种子会重新写入
```

启动日志会明确告诉你是哪条路径：

```
agent_seed.applied agent_id=demo model=deepseek-flash tools=["echo","http_get","write_file"]
agent_seed.skipped agent_id=demo (already in the store)
```

整个运行时状态归零：`redis-cli -n 0 FLUSHDB`。

---

## 10. 设计口径

来自 `Agent中台.md`，用于对齐「这个数字怎么算」：

- **接口级可用性 vs 业务成功率**：99.9% 是接口级口径；模型抖动、MCP 超时若返回了
  降级事件，算接口可用
- **单轮决策 P95 3-6s**：包含 `before_model → llm → after_model → tool → checkpoint 写入`
- **工具调用成功率 98%**：含自动重试；首次失败 5-7%，靠重试挽回
- **长程任务中断恢复 99%**：按发生中断的任务统计；1% 失败主要是沙箱回收、外部工具已
  不可用、Checkpoint 版本不兼容

## 11. 后续路线

```
MVP (当前)
  └─ 沙箱机制（业务容器 + secguard + 凭证代理）
      └─ Agent 实例热加载（Map + 队列 + 两线程轮询）
          └─ MCP 客户端
              └─ MySQL / 对象存储 / 多层存储
                  └─ 真实多供应商 + 接入层鉴权
                      └─ 灰度 + 双机房
```

已知的技术债（都不是 bug，是明确欠着的）：

- **平台无登录体系**，`user_id` 是调用方自报的字符串。它挡住了跨用户的**误继承**，
  挡不住**伪造身份** —— 真正的认证属于接入层
- `hitl` 没有超时自动 reject；挂起的 thread 会一直等到 Checkpoint TTL 到期
- 敏感工具集只能通过代码 `Tool.sensitive` 或配置追加，**配置无法解除**（刻意的）
