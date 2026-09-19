"""Which natural-language phrasings actually trigger the sensitive tool?

The walkthrough uses "Use the write_file tool to save X into notes/y.txt",
which is not how anyone talks. This probes how far ordinary phrasing gets,
against the real DeepSeek model, and whether the system prompt is the lever.

For each prompt we care about one thing: did the model reach for write_file
(and therefore hit the approval gate), or did it just answer in prose? A prose
answer that claims the file was saved is the worst outcome — the user believes
something happened that did not.

Runs in-process against the real API with fakeredis, so it never touches your
Redis or writes into the repo. Costs one LLM call per prompt per variant.

    env -u http_proxy -u https_proxy .venv/bin/python scripts/probe_natural_language.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import fakeredis.aioredis

from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.config import Settings
from agent_platform.core.agent_loop import AgentLoop, HITLInterrupt
from agent_platform.core.providers import create_chat_model
from agent_platform.tools import build_default_registry

# The system prompt the demo agent ships with today.
WEAK = (
    "You are a helpful assistant. Use the available tools when the user "
    "asks you to."
)

# Same agent, with the tool-use policy actually spelled out. Only the prompt
# differs — model, temperature and tools are identical.
STRONG = """You are a helpful assistant with access to tools.

Use the write_file tool whenever the user asks you to remember, save, record,
note down, jot down, write or store anything — even when they never say the
word "file" and never give a path. Choose a sensible path under notes/
yourself, and tell the user which path you used.

Never say that something was saved, written or recorded unless you actually
called the tool."""

# (group, prompt). Chinese first: that is how this platform's users talk.
PROMPTS: list[tuple[str, str]] = [
    ("点名工具+路径", 'Use the write_file tool to save "hello" into notes/a.txt'),
    ("意图明确+有路径", "帮我把「明天10点开会」写到 notes/meeting.md 里"),
    ("意图明确+有路径", "把 hello world 保存到 notes/hello.txt"),
    ("意图明确+有路径", 'Save "buy milk" to notes/todo.txt'),
    ("意图明确+无路径", "帮我把「明天10点开会」记到文件里"),
    ("意图明确+无路径", "把这段话保存成文件：今天天气不错"),
    ("意图明确+无路径", "Create a file containing the text hello"),
    ("意图隐含", "记一下，明天10点开会"),
    ("意图隐含", "帮我存个档：项目下周交付"),
    ("意图隐含", "save this for me: buy milk"),
    ("纯口语", "明天10点要开会，别忘了"),
    ("纯口语", "Can you jot this down? tomorrow 10am meeting"),
]


async def probe(loop: AgentLoop, prompt: str, thread_id: str) -> tuple[str, str]:
    """Return (outcome, detail) for one prompt."""
    said: list[str] = []
    try:
        async for ev in loop.run(
            thread_id=thread_id, user_id="probe", user_message=prompt
        ):
            if ev.type == "hitl_required":
                args = ev.data.get("tool_arguments") or {}
                return "HITL", f"path={args.get('path')}"
            if ev.type == "tool_call":
                names = [c["name"] for c in ev.data.get("calls", [])]
                return "OTHER_TOOL", ",".join(names)
            if ev.type == "assistant" and ev.data.get("content"):
                said.append(ev.data["content"])
            if ev.type == "error":
                return "ERROR", str(ev.data.get("message"))[:60]
    except HITLInterrupt as e:  # pragma: no cover - defensive
        return "HITL", e.approval_id
    return "NO_TOOL", (said[-1][:70] if said else "(empty)")


async def run_variant(
    settings: Settings, system_prompt: str, label: str
) -> list[tuple[str, str, str]]:
    redis = fakeredis.aioredis.FakeRedis()
    ckpt = CheckpointStore(redis, ttl_seconds=300)
    out: list[tuple[str, str, str]] = []
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    for i, (group, prompt) in enumerate(PROMPTS):
        loop = AgentLoop(
            llm=create_chat_model("deepseek-flash", settings),
            tools=build_default_registry(settings),
            checkpoint=ckpt,
            system_prompt=system_prompt,
            settings=settings,
            agent_id="nl-probe",
            config={
                "tools": ["echo", "http_get", "write_file"],
                "sensitive_tools": ["write_file"],
            },
        )
        outcome, detail = await probe(loop, prompt, f"{label}-{i}")
        mark = "OK  " if outcome == "HITL" else "miss"
        print(f"  {mark} [{group:<10}] {prompt[:44]:<46} -> {outcome} {detail}")
        out.append((group, prompt, outcome))
    await redis.aclose()
    return out


def summarize(label: str, results: list[tuple[str, str, str]]) -> None:
    hits = sum(1 for _, _, o in results if o == "HITL")
    print(f"\n  {label}: {hits}/{len(results)} 触发了 HITL")
    by_group: dict[str, list[int]] = {}
    for group, _, outcome in results:
        by_group.setdefault(group, []).append(1 if outcome == "HITL" else 0)
    for group, marks in by_group.items():
        print(f"    {group:<12} {sum(marks)}/{len(marks)}")


async def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="nl-probe-"))
    settings = Settings(
        tool_write_file_root=str(workdir),
        llm_max_retries=1,
        llm_retry_base_delay=0.0,
    )
    if settings.deepseek_key_source() == "unset":
        print("No DeepSeek key configured — nothing to probe against.")
        return

    weak = await run_variant(settings, WEAK, "A. 现在的 system prompt")
    strong = await run_variant(settings, STRONG, "B. 把工具使用策略写清楚的 system prompt")

    print(f"\n{'=' * 78}\n结论\n{'=' * 78}")
    summarize("A 现状", weak)
    summarize("B 改进后", strong)


if __name__ == "__main__":
    asyncio.run(main())
