"""Agent instance manager — the MVP stand-in for the hot-reload system.

Per Agent中台.md "# Agent实例" / "## 2.热加载":
- `agentMap: {agent_id -> AgentLoop}` is the live instance registry
- A separate thread polls config every 90s and re-instantiates changed agents
- A separate thread drains the retired-instance queue every 60s

The MVP only implements the `agentMap` half — the polling threads are stubbed
in `agent_platform.agent_manager.AgentManager` and left as TODOs.
"""

from agent_platform.store.agent_manager import AgentManager

__all__ = ["AgentManager"]
