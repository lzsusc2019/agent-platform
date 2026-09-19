"""Domain layer: the agent itself, with no knowledge of storage or HTTP.

Message and ToolCall models, the ReAct loop, the checkpoint schema, the
ChatModel port, the Tool abstraction. Import freely from here; import nothing
from infra/ or api/, which is what keeps the loop testable without Redis.
"""
