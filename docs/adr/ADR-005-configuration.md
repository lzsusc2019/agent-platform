# ADR-005: 配置分层，代码里不留字面量

- 状态：已采纳（第二版，引入 YAML 归档层）

## 背景

第一版把不少可调参数直接写在了使用它们的代码旁边：

```python
# 反面教材，曾经真实存在于代码里
async with httpx.AsyncClient(timeout=5.0) as client:      # tools/builtins.py
    return f"status={r.status_code} body={r.text[:500]}"

max_tokens: int = 4096                                    # infra/config_store.py
ttl_seconds: int = 30 * 60                                # infra/checkpoint_store.py
base_url: str = "https://api.deepseek.com"                # infra/providers.py
agent_id="demo", tools=["echo", "http_get"]               # infra/config_store.py
```

问题有三个：**不可运维**（改超时要改代码）、**不可审计**（环境实际值说不清）、
**测试难写**（只能 monkeypatch）。

把这些提成 `Settings` 之后又暴露了第二个问题：**环境变量不适合表达结构化配置**。
agent 定义有工具列表、多行提示词、嵌套 metadata，写成
`AGENT_PLATFORM_DEMO_AGENT_TOOLS='["echo","http_get"]'` 既难写又难 review，
而且它们应该是**归档在仓库里、能 diff 的**东西。

## 决策

配置分四层，优先级由低到高：

```
1. 代码默认值                 src/agent_platform/config.py
2. config/platform.yaml       归档的、可 review 的平台配置（进仓库，含本部署的 key）
3. config/platform.local.yaml 可选的每机覆盖（.gitignore 掉）
4. 环境变量 / .env            AGENT_PLATFORM_*

另外：Settings(**kwargs) 显式传参压过以上全部 —— 测试靠这层钉住行为。
```

### 密钥放哪一层

当前部署把 key 写在 `config/platform.yaml`，启动即注入。这是**有意为之**：
这个仓库不公开，而把 key 只放环境变量有个现实问题——每次开新 shell 都要
`export`，忘了就以为"配置没生效"。

同时保留 `config/platform.local.yaml`（gitignored）作为逃生口：

- 优先级高于 `platform.yaml`，低于环境变量
- 仓库将来要公开/共享时，把 key 那一行挪过去即可，其它不用动
- 每个开发者也可以用它放自己的 key，不影响别人的检出

**没有**加"仓库文件含密钥就报警"的检查：既然选择了把 key 放在归档文件里，
那条警告只会变成每次启动的噪音。约束写在这里，写在 `platform.yaml` 的注释里，
而不是写成运行时的告警。

确实保留的保护：

- `.gitignore` 仍覆盖 `config/platform.local.yaml` 和 `.env`
- key 前后空格自动 trim；含全角字符或换行在**写入时**就被 400 拦下
- 启动日志打印 key 的**来源层**和打码值，回答"到底注入了没有"
- 测试通过 `AGENT_PLATFORM_PLATFORM_YAML_FILE` 指向临时空文件，不读开发者的本地配置

```
INFO config.llm provider=DeepSeekChatModel model=deepseek-flash \
     key=sk-46fc...0f51 source=config/platform.yaml
```

### 密钥的写法兼容

| 来源 | 写法 |
|---|---|
| YAML / kwargs | `deepseek_api_key`，别名 `deepseek_apikey` |
| 环境变量 | `AGENT_PLATFORM_DEEPSEEK_API_KEY` / `..._APIKEY` |
| 环境变量回退 | `DEEPSEEK_API_KEY`（官方 SDK 的变量名） |

别名用 `@model_validator(mode="before")` 归一化。环境变量别名不能在同一个校验器里做
（`validation_alias` 会绕过 `AGENT_PLATFORM_` 前缀，破坏标准字段名的查找），所以
放在 `mode="after"` 里直接读 `os.environ`。

Agent 定义不在 `Settings` 里，而在 `config/agents.yaml`：

```yaml
agents:
  - agent_id: demo
    model: deepseek-flash
    system_prompt: |
      You are a helpful assistant.
    tools: [echo, http_get, write_file]
    sensitive_tools: [write_file]
```

这两个字段的语义见 README 的「工具策略」和 ADR-004 的「什么算敏感」。一句话版本：
`tools` 是白名单且**留空等于不限制**（既有配置全靠这条才没被缴械），`sensitive_tools`
**只能加不能减**（不能靠配置关掉代码里声明好的审批闸门）；白名单是执行期强制的，
不只是"不发给模型看"。

启动时按 agent_id 逐条写入运行时存储，**已存在的跳过** —— 所以 Dashboard 上
改过的 agent 不会被重启冲掉。文件路径由 `AGENT_PLATFORM_AGENT_SEED_FILE` 指定，
设为空串即关闭种子。

### 为什么标量进 yaml、密钥进 env

| 类型 | 去处 | 理由 |
|---|---|---|
| 超时、TTL、阈值、重试次数 | `platform.yaml` | 可以 diff、可以 review |
| agent 定义（列表 / 多行文本 / 嵌套） | `agents.yaml` | 结构化，env 表达不了 |
| 密钥、每台机器不同的 endpoint | 环境变量 | **绝不能进仓库** |

### 该进配置 / 该留代码的分界线

| 该进配置 | 该留在代码里 |
|---|---|
| 超时、重试次数、TTL、阈值 | 算法内部的常数（如整除语义） |
| 模型名、endpoint、路径 | 协议字段名（如 `choices`） |
| agent 定义的每个字段 | 数据结构定义 |
| 默认提示词、截断长度、协议白名单 | 测试替身的行为 |

### 配套约束

1. **没有静默默认值。** `DeepSeekChatModel` 的 `base_url`/`model`/`timeout`
   都是必填，由 `create_chat_model` 从 `Settings` 传入。模型名只有一个来源，
   配错了会立刻报错，而不是悄悄用一个库默认值兜住。

2. **构造器不吃默认值。** `CheckpointStore(redis)` 必须显式传 `ttl_seconds`。

3. **失败要响。** `agents.yaml` 语法错 / 字段类型错 → 抛 `AgentSeedError`，
   拒绝启动。带着缺失的 agent 静默起来更糟。文件不存在则只是警告 + 跳过。

4. **归档不能漂移。** `resource/tests/test_configurability.py` 断言 `platform.yaml`
   里 pin 的每个值都等于当前代码默认值 —— 改了一边没改另一边就红。

5. **行为测试锁住接线。** 把字面量提成配置只做对了一半，值还得真的送达。
   同一文件逐个断言每项设置作用于对应代码路径（例如读 `redis.ttl()` 验证 TTL）。

## 后果

- ✅ `cli config` 打印全部生效配置（`Settings.redacted()` 脱敏密钥）
- ✅ `config/platform.yaml` 和 `config/agents.yaml` 在仓库里可 diff、可 review
- ✅ 测试用 `Settings(...)` 或临时 YAML 精确控制被测行为
- ✅ 凭证在**写入时**就校验（见下），不是等发请求才炸
- ⚠️ `platform.yaml` 是 CWD 相对路径；换目录启动要设绝对路径
- ⚠️ 三层优先级需要在文档里讲清楚，否则容易困惑"为什么改了 yaml 没生效"

## 附：为什么加了 `validate_api_key`

一个 API key 里混进全角字符（复制粘贴时很常见），httpx 会在编码
`Authorization` header 时抛：

```
UnicodeEncodeError: 'ascii' codec can't encode characters in position 43-45
```

这条信息既没说哪个字段有问题，也没说是什么问题，最后还被包成
`http 400 failed to build agent ...` 吐给前端。现在三道防线：

1. **写入时**（Dashboard 保存 / `PUT /admin/api/secrets`）→ 400 + 人话
2. **读取时**（`AgentManager._resolve_provider_key`）→ 兜住历史脏数据
3. **构造时**（`DeepSeekChatModel.__init__`）→ 最后一道，防直接用它的人

报错长这样：

```
API key contains non-ASCII characters at position 6-6 ('：'). Keys are ASCII;
this is usually a full-width character (e.g. a full-width colon) or a smart
quote picked up during copy-paste.
```

## 与原文的对应

`Agent中台.md` 里这些值都是经验值（30 分钟 TTL、100 轮上限、140K 压缩阈值、
单轮 3–6 秒）。经验值天然因环境而异；把它做成可归档、可覆盖的配置，才能在不同
规模的部署里重新校准，也才说得清"这个环境实际跑的是什么"。
