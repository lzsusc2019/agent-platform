# ADR-003: Checkpoint 存储模型与幂等键

## 背景

原文 Checkpoint 是对话级状态持久化，Key 是 Agent 实例会话 ID，Value 是状态快照
（messages、last_config、checkpoint_id），TTL 30 分钟。核心问题：
**工具执行成功但 Checkpoint 写失败，怎么不重复执行？**

## 决策

### Key 设计

```
ckpt:{thread_id} -> CheckpointSnapshot (JSON, TTL 30m)
tool:{idempotency_key} -> ToolExecutionRecord (TTL 30m, 仅 debug 用)
```

### Snapshot 结构

```python
class CheckpointSnapshot(BaseModel):
    thread_id: str
    version: int              # 兼容用：升级后旧快照可识别
    messages: list[Message]   # 完整 ReAct 状态
    pending_tools: dict[str, ToolPendingState]  # tool_call_id -> 状态
    status: Literal["running", "waiting_approval", "finished"]
    last_config: dict         # 配置快照，用于恢复时比对是否仍然有效
```

### 幂等键 + 三状态

每个 tool_call 都生成全局唯一的 `idempotency_key`，在执行前写 `PENDING`，执行完写 `DONE`，
执行失败写 `FAILED`。恢复时：

- `DONE` → 复用上次结果
- `PENDING` → 未知态，**不能当失败重试**，要查业务库或等待外部确认
- `FAILED` → 可重试（但要尊重下游系统的幂等约束）

### 并发一致性

原文用 IP 哈希路由到同节点 + Redis 单线程串行化。MVP 范围：
**接受 Redis 是单写入源**（Redis 单线程命令队列已天然串行化），不做 IP 哈希。

## 后果

- ✅ 三状态机明确，避免"失败就重试"的踩坑
- ✅ version 字段支持 Checkpoint 兼容升级
- ⚠️ 单进程 Redis 是单写入源假设，集群部署需要再评估
