# ADR-001: MVP 范围与边界

- 状态：已采纳
- 日期：MVP 起步日

## 背景

`Agent中台.md` 描述了一个生产级的中台，涉及 5 层架构 + 沙箱 + 热加载 + MCP + 4 类存储 + 完整观测。
一次提交显然吃不下。

## 决策

MVP 只做主路径（Agent Loop + Checkpoint + HITL + SSE），其它模块留接口和 TODO：

| 模块                  | MVP | 后续 |
|-----------------------|-----|------|
| Agent Loop / ReAct    | ✅  | —    |
| 上下文压缩            | ✅（同步摘要）| 异步滚动 / 卸载 |
| LLM 重试 + 空响应     | ✅  | —    |
| 100 轮熔断            | ✅  | —    |
| Checkpoint (Redis)    | ✅  | —    |
| 幂等键 + PENDING      | ✅  | —    |
| HITL Interrupt        | ✅  | —    |
| SSE 流式              | ✅  | —    |
| 本地工具注册          | ✅（demo）| MCP 接入 |
| FastAPI 接入层        | ✅（限流/审计占位）| 鉴权/租户 |
| 沙箱                  | ⏳ TODO | 双容器 + secguard |
| Agent 实例热加载      | ⏳ TODO | Map + 队列 |
| MCP 客户端            | ⏳ TODO | MCP 协议 |
| MySQL / OBS           | ⏳ TODO | 多层存储 |
| 真实 LLM 供应商       | ⏳ TODO | 默认 mock，可注入 |

## 后果

- ✅ 跑通核心对话 + 中断恢复主路径，单元测试可覆盖
- ✅ 测试不依赖网络 / Redis（用 fakeredis + mock LLM）
- ⚠️ 默认配置下不抗真实 LLM 抖动（这是后续 Adapter 层的事）
- ⚠️ 没有沙箱意味着工具代码直接 in-process 执行（仅 demo 工具安全）
