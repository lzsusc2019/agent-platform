# Agent Platform (灵枢智能体中台 · MVP)

> 一个可恢复、可中断、可热更新的智能体中台 MVP 骨架

这是把 `Agent中台.md` 里的设计落到代码的第一版。范围刻意收紧 ——
跑通 **Agent Loop + Checkpoint + HITL** 这条主路径，其它模块（沙箱、热加载、MCP）
留接口和 TODO，后续按 ADR 增量落地。

附带的 **Admin Dashboard** 在 `/admin/` 下，是一个单页前端，做 Agent 配置
查看/编辑、Checkpoint 检视、HITL 审批调试、调试 chat 发送。

---

## 0. Admin Dashboard

启动后访问 <http://localhost:8000/admin/> 即可。提供 6 个 tab：

- **Agents** — 列出所有 `agent_id`，左侧编辑表单（system_prompt、model、temperature、max_tokens、tools、sensitive_tools、skills）。保存后会让 `AgentManager` 失效缓存，下一次 chat 会重建 AgentLoop 读最新配置。`tools` / `sensitive_tools` 是真正的策略，见「工具策略」
- **Debug Chat** — 直接发送 SSE chat，输入 agent_id、content，可选 thread_id（空则自动生成），approval_id（用于从 HITL 挂起恢复）。事件流按颜色高亮实时打印。触发 HITL 后 `thread_id` 和 `approval_id` 会自动回填，直接点 **Approve & Resume** 就能闭环
- **Checkpoints** — 列出所有 thread_id、状态徽章（running / waiting_approval / finished），点击查看完整 snapshot JSON
- **HITL Pending** — 列出所有等待审批的 thread + 触发工具名 + 每行的 **approve / reject** 按钮。批准后自动跳到 Debug Chat 并立即恢复，直接看到工具执行和最终回答
- **Tools** — 列出当前注册的所有工具，敏感工具标红
- **Providers** — 配 DeepSeek API key 的地方。输入 key → 保存 → 下一次 chat 即生效（**无需重启**），AgentManager 自动 invalidate 受影响的缓存 loop。带 "test" 按钮直接 ping `/models` 验连通性

### 配 DeepSeek key（不用重启）

1. 启动服务（`AGENT_PLATFORM_USE_FAKE_REDIS=true .venv/bin/python -m agent_platform.cli serve --reload`）
2. 配 key：写进 `config/platform.yaml`（见下方「注入 DeepSeek API key」），或启动后浏览器打开 <http://127.0.0.1:8000/admin/> → **Providers** tab → 在 deepseek / api_key 行点 "set"
3. 用 Dashboard 配的话，点 "test" 可以验连通性；用文件配的话启动日志会打印 key 来源
4. 切到 **Debug Chat** 选 `demo`，发条消息。`start` 事件里 `llm` 字段会显示 `DeepSeekChatModel`（如果是 `MockChatModel` 说明 key 没生效）
5. 事件流里的 `assistant` 内容就是真 DeepSeek 响应；失败会以 `error` 事件显示原因，不会静默断流

Dashboard 不会显示完整 key，只显示 `sk-********xxxx` 这种遮罩形式。

API 都在 `/admin/api/*` 下（新增的 secrets API）：

```bash
# 设置 DeepSeek key（store 优先于 env）
curl -X PUT http://localhost:8000/admin/api/secrets/deepseek/api_key \
  -H 'content-type: application/json' \
  -d '{"value":"sk-xxx","note":"primary"}'

# 测试连通性（真打 /models）
curl -X POST http://localhost:8000/admin/api/secrets/deepseek/api_key/test
```

- **Agents** — 列出所有 `agent_id`，左侧编辑表单（system_prompt、model、temperature、max_tokens、tools、sensitive_tools、skills）。保存后会让 `AgentManager` 失效缓存，下一次 chat 会重建 AgentLoop 读最新配置
- **Debug Chat** — 直接发送 SSE chat，输入 agent_id、content，可选 thread_id（空则自动生成），approval_id（用于从 HITL 挂起恢复）。事件流按颜色高亮实时打印
- **Checkpoints** — 列出所有 thread_id、状态徽章（running / waiting_approval / finished），点击查看完整 snapshot JSON
- **HITL Pending** — 列出所有等待审批的 thread + 触发工具名 + 每行的 **approve / reject**；下方 **Active approvals** 列出当前生效的授权（工具 / 目标 / 剩余时效），可逐条 revoke
- **Tools** — 列出当前注册的所有工具，敏感工具标红

API 都在 `/admin/api/*` 下，方便 curl 调：

```bash
# 列出所有配置
curl http://localhost:8000/admin/api/configs

# 编辑/创建
curl -X PUT http://localhost:8000/admin/api/configs/qa \
  -H 'content-type: application/json' \
  -d '{"system_prompt":"you are a QA bot","tools":["echo"],"sensitive_tools":[]}'

# 调试 chat (SSE)
curl -N -X POST http://localhost:8000/admin/api/chat \
  -H 'content-type: application/json' \
  -d '{"agent_id":"qa","content":"please echo hello"}'

# 看 Checkpoint
curl http://localhost:8000/admin/api/checkpoints/<thread_id>
```

> Dashboard 无鉴权 — 仅供本地开发 / 内网 debug 使用，不要直接暴露到公网。

---

## 1. 范围（MVP 包含）

- ✅ **Agent Loop**：ReAct 推理循环、上下文阈值压缩、LLM 重试（指数退避）、空响应兜底、100 轮硬熔断
- ✅ **Checkpoint**：Redis 持久化运行时状态（messages、last_config、tool idempotency keys），TTL 30 分钟
- ✅ **HITL**：敏感工具 → Interrupt 挂起 → SSE 通知 → 审批通过从 Checkpoint 恢复 → 继续 Loop
- ✅ **SSE 流式输出**：`message` / `tool_call` / `tool_result` / `hitl_required` / `finish` / `error` 事件
- ✅ **本地工具注册中心**：`echo`、`http_get` + 一个标记为敏感的 `write_file`（受限目录内写文件，需审批）
- ✅ **接入层**：FastAPI 路由（会话、聊天 HITL 审批接口）+ 限流/审计中间件占位
- ⏳ **沙箱 / 双容器 / 凭证代理**：接口预留，TODO
- ⏳ **Agent 实例热加载 / Map + 队列**：接口预留，TODO
- ⏳ **MCP 客户端**：接口预留，TODO

## 2. 不在 MVP 范围（明确边界）

- 真实 LLM 供应商适配（默认用可注入的 mock 模型，跑测试无需网络）
- MySQL / OBS 真实集成（Checkpoint 只走 Redis；消息/会话/Skill 持久化留 TODO）
- 鉴权 / 租户 / 多租户隔离（中间件占位）
- Dashboard / 管理后台
- 灰度发布 / 双机房容灾

## 3. 架构一张图

```
┌──────────────┐  SSE   ┌──────────────┐
│   Frontend   │<------>│  FastAPI     │
└──────────────┘        │  (api/)      │
                        └──────┬───────┘
                               │
                        ┌──────▼──────────────────────┐
                        │     Agent Engine            │
                        │   (core/agent_loop.py)      │
                        │                             │
                        │  ┌──────────────────────┐   │
                        │  │ before_model         │   │
                        │  │   • 上下文阈值压缩    │   │
                        │  │   • token 估算        │   │
                        │  ├──────────────────────┤   │
                        │  │ llm_call             │   │
                        │  │   • 指数退避重试      │   │
                        │  │   • 空响应兜底        │   │
                        │  ├──────────────────────┤   │
                        │  │ after_model          │   │
                        │  │   • 敏感工具检测      │   │
                        │  │   • HITL Interrupt    │   │
                        │  ├──────────────────────┤   │
                        │  │ tool_executor         │   │
                        │  │   • 幂等键 + PENDING  │   │
                        │  │   • asyncio.gather    │   │
                        │  └──────────────────────┘   │
                        └──────┬──────────┬───────────┘
                               │          │
                       ┌───────▼────┐ ┌────▼────────┐
                       │Checkpoint  │ │  Tool       │
                       │(Redis)     │ │  Registry   │
                       └────────────┘ └─────────────┘
```

## 4. 关键决策

详见 [`docs/adr/`](docs/adr/)：

- `ADR-001` MVP 范围与边界
- `ADR-002` 为什么 ReAct Loop 自己实现而不是完全交给 LangGraph
- `ADR-003` Checkpoint 存储模型与幂等键设计
- `ADR-004` HITL 三状态机与恢复语义
- `ADR-005` 配置分层（代码默认 / platform.yaml / 环境变量），代码里不留字面量

## 5. 跑起来

```bash
# 安装（uv 推荐）
uv sync --extra test

# 跑测试（不需要 Redis，使用 fakeredis）
uv run pytest

# 启动服务（默认 mock LLM + 真 Redis）
uv run agent-platform serve --reload
```

### 没有 Redis？

开发环境可以用内存版 fakeredis（数据每次重启清空）：

```bash
AGENT_PLATFORM_USE_FAKE_REDIS=true .venv/bin/python -m agent_platform.cli serve
```

启动时会打印警告：`use_fake_redis=true: data is in-memory only, lost on restart. Do NOT use this in production.`

### 在 PyCharm / VS Code 里断点调试

新建一个 **Python** Run/Debug Configuration，填三个字段就够：

| 字段 | 值 |
|---|---|
| Python interpreter | `<项目根>/.venv/bin/python` |
| **Module name**（不是 Script path） | `agent_platform.cli` |
| Parameters | `serve --host 127.0.0.1 --port 8000` |
| Working directory | `<项目根>`（含 `pyproject.toml` 的那层） |

然后直接按 Debug。**不要勾 `--reload`。**

> `--reload` 是 PyCharm 调试的头号坑。uvicorn 的 reloader 会 fork 一个子进程来跑真正的
> app，而调试器挂在父进程上 —— 结果是服务正常跑、断点一个都不命中，人很容易误以为是
> "断点位置不对"而白白排查半小时。改代码后手动重启（PyCharm 的 Rerun 按钮）即可。

如果想更贴近生产入口，也可以直接调 uvicorn（同样不要 `--reload`）：

| 字段 | 值 |
|---|---|
| Module name | `uvicorn` |
| Parameters | `agent_platform.cli:make_app --factory --host 127.0.0.1 --port 8000` |

**跑测试**再建一个：Module name `pytest`，Parameters `-q`，工作目录同上。

#### 工作目录别填错

所有配置入口都是相对路径（`config/platform.yaml`、`config/agents.yaml`）。填错工作目录
的后果不是报错，而是**静默降级成 MockChatModel**：agent 照常回话，只是回的是 mock 文本，
日志里只有一行 WARNING。PyCharm 的默认工作目录是 content root，如果你把
`Documents/personal` 整个作为项目打开，默认值就是错的。

`resolve_config_path()` 现在会在 CWD 找不到时回退到项目根（从源码位置推导），所以填错也
能跑对；但显式填对更省心。

#### 几个值得下断点的位置

| 位置 | 看什么 |
|---|---|
| `core/agent_loop.py` → `run()` | 整个 ReAct 循环、恢复分支、100 轮熔断 |
| `core/agent_loop.py` → sensitive-tool 检测处 | 哪一次 tool call 触发了 HITL |
| `core/providers.py` → `DeepSeekChatModel.ainvoke()` | **真实出站 payload**，排查 400/422 必看 |
| `tools/builtins.py` → `WriteFileTool.run()` | 审批放行后的实际写盘，断在这里能确认路径校验 |
| `store/agent_manager.py` → `_resolve_llm()` | 这次到底解析成了哪个 ChatModel |

开始调试前先在 Run Configuration 的环境变量里加上 `AGENT_PLATFORM_LOG_LEVEL=DEBUG`。

#### 一个会让调试结果完全失真的坑：Redis 里的陈旧 agent

`seed_defaults()` 只在 `agent_id` **不存在**时写入，这样设计是为了不冲掉 Dashboard 上的
改动。代价是：你改了 `config/agents.yaml` 之后重启，**已存在的 agent 不会被更新**。

典型症状是改了 seed 却毫无反应，chat 一直回 mock 的 `ok`。清掉重来：

```bash
curl -X DELETE http://127.0.0.1:8000/admin/api/configs/demo   # 或 Dashboard 的 Agents tab 删除
# 然后重启服务，种子会重新写入
```

启动日志会明确告诉你是哪条路径：

```
agent_seed.applied agent_id=demo model=deepseek-flash tools=['echo', 'http_get', 'write_file']
agent_seed.skipped agent_id=demo (already in the store)
```

想让整个运行时状态归零：`redis-cli -n 0 FLUSHDB`。

## 配置分层

平台**不硬编码任何可调参数**，配置分四层，优先级由低到高：

```
1. 代码默认值                 src/agent_platform/config.py
2. config/platform.yaml       归档的、可 review 的平台配置（进仓库）
3. config/platform.local.yaml 可选的每机覆盖（.gitignore 掉）
4. 环境变量 / .env            AGENT_PLATFORM_*
```

另加 `Settings(**kwargs)` 显式传参压过全部（测试用这层）。

```bash
# 查看当前生效值（密钥自动脱敏）
.venv/bin/python -m agent_platform.cli config
```

### 注入 DeepSeek API key

写进 `config/platform.yaml`，启动即注入：

```yaml
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
| 环境变量回退 | `DEEPSEEK_API_KEY`（官方 SDK 的变量名，已导出的话不用再配） |

启动时会打印一行，告诉你 key 从哪来、值是什么（打码）：

```
INFO config.llm provider=DeepSeekChatModel model=deepseek-flash \
     key=sk-46fc...0f51 source=config/platform.yaml
```

**卫生习惯**：

- `config/platform.yaml` 是进仓库的文件。当前部署把 key 放在这里；**如果仓库将来要公开或共享**，把这一行挪到 `config/platform.local.yaml`（已在 `.gitignore` 里）或环境变量，两个位置优先级都高于 `platform.yaml`
- key 前后空格自动 trim；含全角字符或换行会在**保存时**就被 400 拦下，不用等到发请求
- `tests/` 里的用例通过 `AGENT_PLATFORM_PLATFORM_YAML_FILE` 指向临时空文件，**不会读你本地的 key**

### config/platform.yaml — 平台配置归档

仓库里 `config/platform.yaml` 是**可 diff、可 review 的配置正本**。只 pin 你确实需要
偏离代码默认值的项——省略的键会继续跟随代码默认值演进。改动示例：

```yaml
max_turns: 50              # 这个环境任务短，收紧到 50
deepseek_model: deepseek-flash
tool_http_get_timeout: 3.0 # 内网快，超时压短
```

> `tests/test_configurability.py` 会断言 YAML 里 pin 的值与代码默认值一致，
> 防止两边悄悄漂移。

### config/agents.yaml — agent 定义归档

agent 定义（工具列表、多行提示词、嵌套 metadata）结构化，塞进环境变量既难写又难 review，
所以单独归档成 YAML：

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
    metadata:
      owner: platform-team
```

- 启动时按 `agent_id` 写入运行时存储，**已存在的跳过** → Dashboard 改过的 agent 不会被重启冲掉
- 省略的字段回落到 `agent_default_*` 设置，最简可以只写 `agent_id` + `model`
- YAML 语法错 / 字段类型错 → 抛 `AgentSeedError` **拒绝启动**（带缺失 agent 静默起来更糟）
- 换文件路径或关闭种子：`AGENT_PLATFORM_AGENT_SEED_FILE=/path/to/agents.yaml` / 设为空串
- `tools` / `sensitive_tools` 的语义见上面的「工具策略」

### 怎么触发并验证 HITL

**触发方式**：让模型去调一个敏感工具就行。`write_file` 在代码里标了 `sensitive`，所以任何让模型写文件的请求都会挂起：

```bash
SID=$(curl -s --noproxy '*' -X POST http://127.0.0.1:8000/v1/sessions \
  -H 'content-type: application/json' \
  -d '{"agent_id":"demo","user_id":"u1"}' | sed 's/.*"thread_id":"\([^"]*\)".*/\1/')

curl -N --noproxy '*' -X POST http://127.0.0.1:8000/v1/sessions/$SID/chat \
  -H 'content-type: application/json' \
  -d '{"content":"Use the write_file tool to save hello into notes/hitl.txt"}'
# -> event: hitl_required   "approval_id": "appr_<thread>_<call>"
```

> 措辞很关键。**要明确点名工具和路径**：含糊的 "write a file" 真模型经常直接回一段话而不调工具。

**完整验证**（含 approve 和 reject 两条分支，以及真正重要的两条断言——批准前工具绝不能执行、驳回后绝不能落盘）：

```bash
scripts/hitl_walkthrough.sh                    # 默认 :8000 / agent demo
```

或者在 Dashboard 里点：**Debug Chat** 发一条写文件的请求 → 看到 `hitl_required`（`thread_id` 和 `approval_id` 会自动回填）→ 点 **Approve & Resume**。也可以在 **HITL Pending** tab 里对任意一条待审批直接点 approve / reject。

不想花真钱就用离线版：`scripts/smoke.py`（mock 模型 + fakeredis，秒级）。

### 一次审批到底批准了什么

「已批准」不是请求里的一个字段，而是 **Redis 里的一条记录**。`/hitl/approve` 写入它，
恢复时 Loop 查它 —— 不看请求里带了什么。

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

### 标量 vs 密钥

| 类型 | 去处 | 理由 |
|---|---|---|
| 超时、TTL、阈值、重试次数 | `platform.yaml` | 可 diff、可 review |
| agent 定义 | `agents.yaml` | 结构化，环境变量表达不了 |
| **密钥**、每机不同的 endpoint | **环境变量** | **绝不能进仓库** |

### 工具策略（按 agent 裁剪）

`AgentConfig` 里有两个字段控制「这个 agent 能用什么工具、哪些要审批」：

| 字段 | 语义 |
|---|---|
| `tools` | 该 agent 可见的工具白名单。**留空 = 不限制**（全部注册工具），不是「禁用所有工具」 |
| `sensitive_tools` | 额外需要 HITL 审批的工具。**只能加，不能减** |

两条语义都是刻意的选择：

- **空 `tools` = 不限制**：所有既有配置和 Dashboard 新建的 agent 这个字段都是空的，解释成「没有工具」会把它们全部悄悄缴械。
- **`sensitive_tools` 只能加不能减**：代码里标了 `sensitive = True` 的工具（`write_file`）永远需要审批，配置漏写或者写空都不会解除。审批闸门不该被一个配置笔误关掉。想给本来安全的工具加审批（比如让 `http_get` 也要批）就用这个字段。

工具白名单是**执行期强制**的，不只是"不给 schema"：schema 列表只是"告知"，被幻觉或被提示注入的模型仍然可以点名一个它没见过的工具，所以 Loop 在真正执行前会再查一次白名单，越权调用会以 `is_error` 的结果回灌给模型（`AgentLoop._execute_tools`）。

还有一个推论：**不被允许的工具不会触发审批弹窗**。既然答案无论如何都是"不"，问人只是噪音，所以直接拒绝。

### 环境变量参考

### 存储

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_PLATFORM_REDIS_URL` | `redis://localhost:6379/0` | Checkpoint / AgentConfig / Secret 后端 |
| `AGENT_PLATFORM_USE_FAKE_REDIS` | `false` | 用 fakeredis 替真 Redis（仅 dev） |
| `AGENT_PLATFORM_CHECKPOINT_TTL_SECONDS` | `1800` | 挂起的 thread 能等多久（秒） |
| `AGENT_PLATFORM_APPROVAL_GRANT_TTL_SECONDS` | `3600` | 一次审批签发后能用多久（秒）。两个时钟互相独立 |

### Agent Loop

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_PLATFORM_MAX_TURNS` | `100` | 单任务 Loop 硬熔断 |
| `AGENT_PLATFORM_LLM_MAX_RETRIES` | `3` | LLM 调用失败重试次数 |
| `AGENT_PLATFORM_LLM_RETRY_BASE_DELAY` | `0.5` | 指数退避基数（秒） |
| `AGENT_PLATFORM_EMPTY_RESPONSE_MAX_RETRIES` | `3` | 连续空响应容忍次数 |

### 上下文压缩

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_PLATFORM_TOKEN_ESTIMATE_CHARS_PER_TOKEN` | `4` | token 估算比（按模型族可调） |
| `AGENT_PLATFORM_CONTEXT_WINDOW_TOKENS` | `260000` | 模型上下文窗口 |
| `AGENT_PLATFORM_COMPRESS_TRIGGER_TOKENS` | `180000` | 压缩触发阈值 |
| `AGENT_PLATFORM_COMPRESS_KEEP_RECENT_TURNS` | `30` | 压缩后保留的原始轮数 |

### Agent 默认值（AgentConfig 缺省时使用）

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_PLATFORM_AGENT_DEFAULT_SYSTEM_PROMPT` | `You are a helpful assistant. Use tools when needed.` | 平台默认提示词 |
| `AGENT_PLATFORM_AGENT_DEFAULT_MODEL` | `mock` | 新建 agent 的默认 model |
| `AGENT_PLATFORM_AGENT_DEFAULT_TEMPERATURE` | `0.7` | 新建 agent 的默认温度 |
| `AGENT_PLATFORM_AGENT_DEFAULT_MAX_TOKENS` | `4096` | 新建 agent 的默认输出上限 |

### Agent 种子归档

agent 定义在 `config/agents.yaml` 里，这里只有一个开关：

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_PLATFORM_AGENT_SEED_FILE` | `config/agents.yaml` | 种子文件路径（CWD 相对）；空串则关闭 |
| `AGENT_PLATFORM_PLATFORM_YAML_FILE` | `config/platform.yaml` | 平台配置归档路径（可挪出源码树） |
| `AGENT_PLATFORM_LOCAL_YAML_FILE` | `config/platform.local.yaml` | 本地覆盖路径 |

### DeepSeek

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_PLATFORM_DEEPSEEK_API_KEY` | `""` | API key（OpenAI 兼容） |
| `AGENT_PLATFORM_DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | endpoint |
| `AGENT_PLATFORM_DEEPSEEK_MODEL` | `deepseek-flash` | 默认模型（`model: "deepseek"` 无后缀时使用） |
| `AGENT_PLATFORM_DEEPSEEK_TIMEOUT` | `30.0` | 单次请求超时（秒） |
| `AGENT_PLATFORM_DEEPSEEK_REQUIRE_KEY` | `false` | key 缺失时是否 fail-fast |
| `AGENT_PLATFORM_DEEPSEEK_CHAT_PATH` | `/chat/completions` | 网关/代理改写用 |
| `AGENT_PLATFORM_DEEPSEEK_MODELS_PATH` | `/models` | Dashboard "test" 按钮用 |

### 工具

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_PLATFORM_TOOL_HTTP_GET_TIMEOUT` | `5.0` | `http_get` 超时（秒） |
| `AGENT_PLATFORM_TOOL_HTTP_GET_MAX_BODY_CHARS` | `500` | 回灌给模型的响应体截断长度 |
| `AGENT_PLATFORM_TOOL_HTTP_GET_ALLOWED_SCHEMES` | `["http://","https://"]` | 协议白名单 |
| `AGENT_PLATFORM_TOOL_WRITE_FILE_ROOT` | `workspace` | `write_file` 可写的根目录，越界路径直接拒绝 |
| `AGENT_PLATFORM_TOOL_WRITE_FILE_MAX_BYTES` | `65536` | 单次写入字节上限（按 UTF-8 字节数算） |

### Admin / 观测

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_PLATFORM_ADMIN_SECRET_TEST_TIMEOUT` | `10.0` | Provider key 连通性检测超时 |
| `AGENT_PLATFORM_LOG_LEVEL` | `INFO` | 日志级别 |

完整列表（含注释）见 `src/agent_platform/config.py`；归档正本见 `config/platform.yaml`
和 `config/agents.yaml`。`tests/test_configurability.py` 断言每项设置真的作用于对应
代码路径且 YAML 与代码默认值不漂移，`tests/test_agent_seed.py` 覆盖种子加载与优先级。

## 6. LLM 供应商配置（DeepSeek / Mock / 自定义）

`AgentConfig.model` 字段决定走哪个供应商，格式是 `"provider[:model_name]"`：

| `model` 字符串 | 行为 |
|---|---|
| `"mock"` | `MockChatModel`，确定性、无网络 |
| `"deepseek"` | DeepSeek，使用 `settings.deepseek_model`（默认 `deepseek-flash`） |
| `"deepseek:deepseek-v4-pro"` | DeepSeek，指定 Pro 模型 |
| `"openai:gpt-4o-mini"` | 报错（未注册）；先 `register_provider("openai", OpenAIChatModel)` 即可 |

### 配置 DeepSeek

1. 去 <https://platform.deepseek.com> 申请 API key
2. 设环境变量：

```bash
export AGENT_PLATFORM_DEEPSEEK_API_KEY=sk-xxx
```

3. 在 Admin Dashboard 把某个 Agent 的 `model` 改成 `deepseek-flash` 并保存。下一次 chat 就会用真 DeepSeek。

> demo agent 的种子默认模型已经是 `deepseek-flash`，所以配好 key 后开箱即用。

或者在 Dashboard 的 Agents tab 编辑器里改 `model` 字段，点保存即可（不需要重启服务）。

### 没设 key 会怎样？

- 默认行为：fallback 到 `MockChatModel`，打印 `WARNING provider.fallback`。Dashboard 仍可调试 chat，但所有 LLM 响应是 mock 的（"ok" / "done: ..."）
- 如果想让缺失 key 立刻 fail（生产建议）：设 `AGENT_PLATFORM_DEEPSEEK_REQUIRE_KEY=true`，启动时会抛 `RuntimeError`

### 加新的供应商（OpenAI、Anthropic、local llama…）

实现 `ChatModel` 接口（看 `src/agent_platform/core/llm.py`）然后在 startup 注册：

```python
from agent_platform.core.providers import register_provider
from agent_platform.core.llm import ChatModel, LLMResponse

class OpenAIChatModel(ChatModel):
    async def ainvoke(self, messages, tools):
        ...
        return LLMResponse(content=..., tool_calls=[...])

register_provider("openai", OpenAIChatModel)
```

之后 `model: "openai:gpt-4o-mini"` 就会被路由到这个实现。深架构参考 `DeepSeekChatModel`（`src/agent_platform/core/providers.py`）—— 整段就 ~80 行，因为 DeepSeek 和 OpenAI 的 API 是兼容的，可以直接 copy 改 base_url + 鉴权 header。

curl 例子：

```bash
# 开个会话
curl -X POST http://localhost:8000/v1/sessions \
  -H 'content-type: application/json' \
  -d '{"agent_id":"demo","user_id":"u1"}'

# 发个普通对话（会触发 echo 工具）
curl -N -X POST http://localhost:8000/v1/sessions/<sid>/chat \
  -H 'content-type: application/json' \
  -d '{"content":"请 echo 一下 hello"}'

# 发个会触发 HITL 的对话（会调用敏感的 write_file）
# SSE 会先推一个 hitl_required 事件并挂起，此时 Checkpoint 已落盘
curl -N -X POST http://localhost:8000/v1/sessions/<sid>/chat \
  -H 'content-type: application/json' \
  -d '{"content":"帮我把这段内容写到 notes/demo.txt"}'

# 审批通过
curl -X POST http://localhost:8000/v1/sessions/<sid>/hitl/approve \
  -H 'content-type: application/json' \
  -d '{"approval_id":"<approval_id>"}'
```

## 7. 验证脚本

三个脚本，按"要不要花真钱"排序：

| 脚本 | 依赖 | 用途 |
|---|---|---|
| `scripts/smoke.py` | 无外部服务 | 端到端跑一遍 HTTP 面：healthz → 建会话 → SSE chat → 工具回合 → HITL 挂起 → 审批 → 恢复 → admin 接口 |
| `scripts/diag_orderings.py` | 无外部服务 | 复现 Dashboard 上的 5 种操作顺序，排查"先点了 A 再点 B"这类问题 |
| `scripts/live_verify.py` | **真 DeepSeek key + 出网** | 打真实 API：工具回合、东莞天气 `http_get`、敏感工具 HITL 全链路，并断言消息顺序合法 |
| `scripts/check_approval_scope.py` | 真 API + 运行中的服务 | 验证「一次审批到底批准了什么」：同文件免审、换文件必审、换用户必审、跳过审批不执行、伪造 id 被拒、撤销后恢复审批 |

```bash
.venv/bin/python scripts/smoke.py
.venv/bin/python scripts/diag_orderings.py
env -u http_proxy -u https_proxy .venv/bin/python scripts/live_verify.py
```

> 本机设了 `http_proxy`，跑 curl 或 live 脚本时记得 `--noproxy '*'` /
> `env -u http_proxy`，否则请求会绕到代理上。

### 一个反复踩到的坑：消息顺序不变式

OpenAI 兼容接口有一条硬规则：

> **带 `tool_calls` 的 assistant 消息，后面必须紧跟应答它每一个 `tool_call_id` 的
> tool 消息，中间不能出现别的角色。**

这条规则有四个地方会踩到，都修过，也都有回归测试：

1. **序列化格式错**：`tool_calls` 少了 `type` 字段、`arguments` 没编码成 JSON 字符串
   → `422 missing field 'type'`（`tests/test_wire_format.py`）
2. **HITL 恢复时空串被当成输入**：恢复请求带 `content: ""`，空串不是 `None`，于是被
   当成真实用户消息插进了 assistant 和 tool 结果之间 → `400 An assistant message
   with 'tool_calls' must be followed by tool messages`
   （`tests/test_hitl_resume_wire.py`）
3. **上下文压缩切断配对**：压缩按条数切分，可能正好切在 assistant 和它的 tool 结果
   中间；`_summarize()` 只保留 user/assistant，被摘要掉的那一半会留下孤儿 tool 消息
   → 同样 400。修法是切分后把孤儿 tool 消息一并推进 `older`
4. **驳回审批后那条调用永远没人应答**：`/hitl/reject` 只把 status 改成 `finished`，
   assistant 的 `tool_calls` 就一直悬着。**驳回本身没问题，坏的是之后**：你再往同一个
   thread 发一句话，这段非法历史被原样重放 → 400

第 2 条还有一个变体：恢复时**同时**带一句真实用户输入，那句话同样会插错位置。所以
Loop 里的用户消息现在是"延后追加"的 —— 先补 tool 结果，再追加用户输入。

#### 兜底：不变式在 Loop 里强制执行

上面四条里，第 4 条是**唯一跑到线上的**，而且它揭示了一个模式：只要每个调用方各自
维护这个不变式，就总会有人漏。所以现在 Loop 在**每次调用模型之前**会重建消息序列
（`repair_tool_call_ordering`）：缺的 tool 回执补在紧跟 assistant 的位置、孤儿 tool
消息丢掉，并打 `loop.repaired_message_history` WARNING。

两个细节值得记：

- **必须"插入"而不是"追加"**。追加看着能修，其实不行 —— provider 要求 tool 回执
  **紧邻**调用，如果中间已经插进了用户消息，追加到末尾等于没修。这是我自己第一版
  兜底代码的错误，被测试抓出来了。
- 打了 WARNING 意味着**别的层出了问题**，值得去看，不要当它是正常路径。

另外 `/admin/api/chat` 现在和 `/v1/sessions/{id}/chat` 一样，对 parked 的 thread 返回
409 而不是默默把用户消息插进悬空调用后面。

## 8. 设计口径提醒

- **接口级可用性 vs 业务成功率**：99.9% 是接口级口径；模型抖动、MCP 超时若返回了降级事件，算接口可用
- **单轮决策 P95 3–6s**：包含 `before_model → llm → after_model → tool → checkpoint 写入`
- **工具调用成功率 98%**：含自动重试；首次失败 5–7%，靠重试挽回
- **长程任务中断恢复 99%**：按发生中断的任务统计；1% 失败主要是沙箱回收、外部工具已不可用、Checkpoint 版本不兼容

## 9. 后续路线

```
MVP (当前)
  └─ 沙箱机制（业务容器 + secguard + 凭证代理）
      └─ Agent 实例热加载（Map + 队列 + 两线程轮询）
          └─ MCP 客户端
              └─ MySQL / OBS / 多层存储
                  └─ 真实 LLM 供应商 + LangGraph 状态机迁移
                      └─ 灰度 + 双机房
```

---

`Agent中台.md` 是这套骨架的源头档案，所有设计取舍在那里都讲过一遍。
