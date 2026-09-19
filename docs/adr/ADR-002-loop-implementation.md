# ADR-002: 为什么 ReAct Loop 自己实现，而不是完全交给 LangGraph

## 背景

LangGraph 是成熟的状态机框架，天然适合 ReAct。但中台对 Loop 有几条"非标"诉求：

1. **Checkpoint 写入时机**要精确可控 —— 每轮工具执行完、HITL 中断、对话完成
2. **HITL 挂起语义**：必须能从 Loop 内部抛 Interrupt，由外部 API 接收审批后，从中断点恢复
3. **幂等键 + PENDING 状态机**需要在 ToolNode 之前/之后插入两步 Checkpoint 写入
4. **可观测性**：每个阶段要 emit 一个结构化事件到 SSE

LangGraph 的 `interrupt()` + `Command(resume=...)` 能做 HITL，但：

- 持久化层是 LangGraph 自己的，替换成我们的 Redis Checkpoint 需要 `BaseCheckpointSaver` 适配
- 工具执行前后插入自定义逻辑要靠 middleware，hook 不如自己写 Loop 直白
- 单元测试要造一个 in-memory 的伪 LangGraph 运行时，复杂度上去了

## 决策

**自己实现 Agent Loop，但保留 LangChain 的 ChatModel / Tool 抽象**，方便后续接
OpenAI / Anthropic 等真实模型。Loop 是一个 async generator，emit 结构化事件，
由 FastAPI 的 SSE 端点消费。

```
agent_loop(state, llm, tools, hitl_guard, ckpt_store, cfg) -> AsyncIterator[LoopEvent]
```

Checkpoint 存储是独立组件 `CheckpointStore`，由 Loop 显式调用。HITL 通过
`Interrupt` 异常从内部 raise，Loop 捕获后状态写入 Checkpoint 后退出，
下次 chat 调用从 Checkpoint 恢复。

## 后果

- ✅ Loop 全流程可读、可测、可调
- ✅ Checkpoint 写入时机精确
- ✅ SSE 事件模型我们自己定义，不被框架绑定
- ⚠️ 重复造了一部分轮子（状态机、条件边）。**后续如果需要复杂多分支 Agent（并行子图、sub-agent）再迁回 LangGraph**，那时用 `BaseCheckpointSaver` 适配我们的存储即可

## 与原文的对应

原文 "主导了用户面架构设计，落地了 Agent Loop、Checkpoint 持久化、HITL、Agent 实例热加载这些模块"
说明 Loop 是用户自己实现的 —— 我们的取舍一致。
