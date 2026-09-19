"""Command-line entry point.

Usage:
    agent-platform serve [--reload] [--port 8000]
    agent-platform check    # in-memory smoke test of the full loop
    agent-platform config   # print the effective configuration
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

import typer
import uvicorn
from fastapi import FastAPI

from agent_platform.agent_seed import load_agent_seeds
from agent_platform.api.app import Runtime, create_app
from agent_platform.config import Settings

app = typer.Typer(help="Agent Platform MVP CLI")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def make_app() -> FastAPI:
    """Build a fresh FastAPI app. Module-level so uvicorn --reload can
    re-import it; called once per worker / per reload cycle.
    """
    settings = Settings()
    _configure_logging(settings.log_level)
    runtime = Runtime.default(settings)
    return create_app(runtime)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on source change."),
) -> None:
    """Run the FastAPI server.

    With --reload, uvicorn requires an import string so the worker can
    re-import the module on file changes. Without --reload we build the
    app in-process and pass the object directly.
    """
    if reload:
        # Import string mode. uvicorn will call agent_platform.cli:make_app
        # after (re)importing this module on every reload.
        uvicorn.run(
            "agent_platform.cli:make_app",
            host=host,
            port=port,
            reload=True,
            factory=True,
        )
    else:
        uvicorn.run(make_app(), host=host, port=port, reload=False)


@app.command()
def config() -> None:
    """Print the effective configuration as JSON.

    Useful for confirming which env vars actually took effect — the
    Dashboard is authoritative for per-agent config, but this is the
    platform-wide view.
    """
    settings = Settings()
    typer.echo(json.dumps(settings.redacted(), indent=2, default=str))


@app.command()
async def check() -> None:
    """Smoke test: run a 2-turn conversation against the real Runtime wiring.

    Uses an in-memory Redis (fakeredis) and the deterministic MockChatModel,
    so it needs no network and no external services. Exercises the same
    Runtime/AgentManager path the server uses.
    """
    settings = Settings(
        use_fake_redis=True,
        checkpoint_ttl_seconds=60,
        llm_max_retries=1,
        llm_retry_base_delay=0.0,
        empty_response_max_retries=1,
    )
    _configure_logging(settings.log_level)
    runtime = Runtime.default(settings)
    await runtime.seed_defaults()

    seeds = load_agent_seeds(settings)
    if not seeds:
        typer.echo(
            f"no agents seeded from {settings.agent_seed_file!r}; nothing to do",
            err=True,
        )
        raise typer.Exit(code=1)
    agent_id, agent_model = seeds[0].agent_id, seeds[0].model
    agent = await runtime.agents.get_or_create(agent_id)

    events: list[dict[str, Any]] = []

    async def drive(content: str) -> None:
        events.clear()
        async for ev in agent.run(
            thread_id="cli-check",
            user_id="cli",
            user_message=content,
        ):
            events.append(ev.model_dump())

    typer.echo(f"--- turn 1: echo (agent={agent_id} model={agent_model}) ---")
    await drive("please echo hello world")
    typer.echo(f"    events: {[e['type'] for e in events]}")

    typer.echo("--- turn 2: write a file -> should request approval ---")
    await drive("write a file please")
    typer.echo(f"    events: {[e['type'] for e in events]}")

    if not any(e["type"] == "hitl_required" for e in events):
        typer.echo("FAIL: expected a hitl_required event", err=True)
        raise typer.Exit(code=1)
    typer.echo("OK")


def main() -> Any:
    """Entry point declared in pyproject.toml [project.scripts]."""
    # `check` is async; route through asyncio.run so `typer` stays simple.
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        asyncio.run(check())
        return
    app()


if __name__ == "__main__":
    main()
