# ADR-004: HITL 三状态机与恢复语义

## 背景

HITL 是 Agent 把"该不该做"的决定权交给人的一次受控中断。原文分 5 步：
触发 → 挂起 → 交互 → 恢复 → 拒绝。

## 决策

### 三状态机

```
running ──(敏感工具未审批)──> waiting_approval
   │                                  │
   │                                  ├─ approve ─> running (从 Checkpoint 恢复)
   │                                  ├─ reject  ─> finished (终止，且补齐 tool 回执)
   │                                  └─ timeout ─> finished (TODO: 后续)
   │
   └─(LLM 无 tool_calls / 100 轮上限 / 工具无权限)──> finished
```

### 什么算「敏感」

两个来源取**并集**：

1. **代码声明**：`Tool.sensitive = True`（`WriteFileTool` 就是）。这是下限。
2. **配置追加**：`AgentConfig.sensitive_tools`，用于给本来安全的工具加审批（例如让 `http_get` 也要人批）。

配置**只能加，不能减**。代码里标了敏感的工具，无论配置怎么写都仍然要审批 —— 审批闸门不应该被一个配置笔误或者一次 Dashboard 误操作关掉。想解除某个工具的审批，改代码是唯一途径，那也意味着它会走一次 review。

推论：**不被允许的工具不触发审批**。`AgentConfig.tools` 是白名单，白名单外的调用直接拒绝（见下），既然答案无论如何都是"不"，弹审批只是噪音。

### Interrupt 协议

Loop 内部检测到敏感工具未审批时：

1. 抛 `HITLInterrupt` 异常，payload 是 `approval_id` + 工具摘要 + 自然语言描述
2. Loop 捕获异常，写 Checkpoint（status=`waiting_approval`）
3. emit `hitl_required` SSE 事件
4. 退出本次 Loop

### 挂起时的落盘

挂起时 Checkpoint 会记录 `approval_id` 和 `status=waiting_approval`。**把 approval_id 存下来而不是让调用方从 thread_id 反推**，是因为它的格式（`appr_{thread}_{tool_call_id}`）是 Loop 的内部约定；admin API 要列出待审批列表并给出可操作的按钮，不该依赖去复刻这个格式。恢复时该字段被清空，否则一个已经跑完的 thread 会永远看起来"可审批"。

### 审批是一次「授权」，不是请求里的一个字段

**这是本 ADR 修订最大的一处。** 早先的实现里，`/hitl/approve` 校验完就返回，**什么都不写**；
Loop 的判断只有一句 `approval_id is None`。于是"已批准"实际等于"resume 请求里带了个
approval_id"，后果有三：

1. **approve 端点是装饰性的** —— 跳过它，直接拿 SSE 里的 approval_id 去 resume，工具照跑
2. **只校验前缀** —— `appr_{thread_id}_随便编` 也能通过
3. **一次批准放行整个回合** —— 只要本次 run 带了 approval_id，这轮里所有敏感调用全部免审

现在授权是一条**记录**，落在 Redis 里，由 `ApprovalStore` 管理：

| 属性 | 语义 |
|---|---|
| **范围** | `Tool.approval_scope(arguments)` 决定一次授权覆盖什么。`write_file` 返回解析后的绝对路径 —— 批准 notes/a.txt **不**授权 notes/b.txt |
| **时效** | `approval_grant_ttl_seconds`，默认 1 小时。过期后同一个文件重新弹审批 |
| **主体** | 授权签发给 `(agent_id, user_id)`，别人的回合不继承 |
| **可审计** | 谁、哪个工具、哪个目标、何时签发，Dashboard 的 Active approvals 面板可见、可撤销 |

范围比较用的是**解析后的路径**，所以 `notes/a.txt`、`./notes/a.txt`、
`notes/../notes/a.txt` 共享同一次授权，而 `../escape.txt` 解析失败返回空 scope，永远不会
匹配到别人合法写入挣来的授权。

### 恢复语义

前端收到事件后调用 `/v1/sessions/{sid}/hitl/approve`（或 `reject`），body 带 `approval_id`。
API 端：

1. 校验 approval_id **精确等于**该 session 正在等待的那一个（不再是前缀匹配）
2. 从 Checkpoint 上记录的 `pending_tool_name` / `pending_tool_arguments` 算出该次调用的
   scope，写入一条授权记录
3. 前端再用 `approval_id` 重启 chat —— Loop **重新查授权**，而不是信这个字段

第 3 步是关键：**恢复时 Loop 不信任请求，只信记录**。所以跳过 approve 直接 resume 会重新
挂起（`reason=grant_missing_or_expired`），授权过期后 resume 也会重新挂起。

> `checkpoint_ttl_seconds` 和 `approval_grant_ttl_seconds` 是两个独立的时钟，没有大小关系：
> 前者约束"挂起的 thread 能等多久"，后者约束"人给出的答复能用多久"。

### 拒绝也必须「应答」

驳回有个不显眼的义务：**被驳回的 tool call 仍然需要一条 tool 回执**。

最早的实现只是把 status 改成 `finished`，assistant 那条带 `tool_calls` 的消息就一直悬着。
驳回当次没有任何异常 —— 坏的是**下一次**：任何人往这个 thread 再发一句话，这段非法历史
被原样重放，provider 回一个 400 `insufficient tool messages following tool_calls`，而报错
指向的是一个跟用户操作毫无关系的消息下标。

所以驳回时会补一条如实说明的 tool 回执（"a human denied approval for this tool call"）。
这不只是为了合规：模型由此**知道自己的请求被拒了**，下一轮会说"刚才那次写入被人工拒绝，
文件没有创建"，而不是莫名其妙地丢掉一个回合。

同样的道理，`/admin/api/chat` 对 parked thread 现在返回 409，而不是把新用户消息插到
悬空调用后面。

### 恢复时的消息顺序不变式

HITL 恢复是本项目最容易破坏 provider 协议契约的地方，单独记一笔。

OpenAI 兼容接口的硬规则：**带 `tool_calls` 的 assistant 消息，后面必须紧跟应答它
每一个 `tool_call_id` 的 tool 消息，中间不能出现别的角色。**

挂起时的 Checkpoint 恰好停在最危险的位置 —— messages 的尾巴是"一个还没有 tool
结果的 assistant 消息"。恢复路径上有三件事会往这个缝里塞东西：

| 踩法 | 症状 | 修法 |
|---|---|---|
| API 传 `content: ""`，空串不是 `None`，被当成真实用户消息 | `400 insufficient tool messages following tool_calls` | `run()` 入口把空白 `user_message` 归一成 `None` |
| 恢复时**同时**带一句真实用户输入，它同样插在 tool 结果之前 | 同上 | 用户消息延后追加：先补 tool 结果，再追加用户输入 |
| 上下文压缩按条数切分，正好切在 assistant 和它的 tool 结果中间 | 同上（长会话才会出现） | `_maybe_compress` 切分后把队首的孤儿 tool 消息推进 `older` |

另外 `hitl_resolved` 事件曾经发两次（通用分支 + 恢复快捷路径各一次），现在只在
恢复快捷路径里发一次，且只在确实有审批在等待时发。

回归测试见 `resource/tests/test_hitl_resume_wire.py`，核心是一个
`assert_valid_openai_order()`：不联网，直接在本地校验出站 payload 不会触发服务端
的那条规则。真实 API 侧的确认由 `scripts/live_verify.py` 负责。

### 不重新审批原则

**HITL 审批的是"意图"，不是"执行结果"**。工具执行失败走容错
（重试 → 回灌模型重规划），不会因为失败就要求重新审批。

而同一次意图在时效内重复出现，也不重复打扰人 —— 但"同一次意图"的粒度是
**目标**，不是"这个回合"：批了 notes/a.txt，接下来一小时里写 a.txt 都不再问，
写 b.txt 立刻再问。早先按回合放行，等于把审批的粒度悄悄放大到了"这个 agent
在这个回合里想干的任何事"。

## 后果

- ✅ 状态机清晰，单一职责
- ✅ 审批结果不耦合工具结果，失败走容错
- ✅ 授权以「目标」为单位、有时效、有主体、可审计可撤销
- ✅ 审批的权威在服务端记录里，客户端无法靠断言一个 id 获得权限
- ⚠️ 恢复语义必须显式维护"消息顺序不变式"，不能想当然地往 messages 尾部 append
- ⚠️ 平台仍无登录体系，"主体"是调用方自报的 `user_id`。它挡住了跨用户的误继承，
  但挡不住伪造身份 —— 真正的身份认证属于接入层，不在 MVP 内
- ⚠️ 暂不做超时自动 reject（TODO）
