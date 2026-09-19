"""Agent Loop — the heart of the platform.

Faithful to Agent中台.md "# Agent loop":
    1. before_model: token budget check + compression
    2. llm_call: invoke ChatModel with exponential-backoff retry + empty-response guard
    3. after_model: tool-call parsing, sensitive-tool detection -> HITL interrupt
    4. tool_executor: PENDING -> execute -> DONE, asyncio.gather for parallel tools
    5. checkpoint write at: end of every tool round, on HITL interrupt, on finish

The Loop is an async generator of LoopEvents. The API layer drains it into SSE.
This is the implementation choice from ADR-002: not LangGraph, because we need
precise control over Checkpoint write points and HITL semantics.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from agent_platform.config.settings import Settings
from agent_platform.domain.checkpoint import (
    CheckpointSnapshot,
    CheckpointStatus,
    ToolPendingState,
)
from agent_platform.domain.errors import (
    HITLInterrupt,
    LoopBudgetExceeded,
    LoopEmptyResponse,
    ToolPermissionDenied,
)
from agent_platform.domain.events import EventType, LoopEvent
from agent_platform.domain.llm import ChatModel
from agent_platform.domain.messages import (
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    repair_tool_call_ordering,
)
from agent_platform.domain.tool import ToolContext, ToolRegistry
from agent_platform.infra.approval_store import ApprovalStore
from agent_platform.infra.checkpoint_store import (
    CheckpointStore,
    new_idempotency_key,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- helpers


def _estimate_tokens(messages: list[Message], chars_per_token: int) -> int:
    """Cheap token estimate: len(text) / chars_per_token.

    A real implementation would call the model's tokenizer. We don't ship one
    in MVP because (a) it bloats deps and (b) the trigger is fuzzy anyway.
    `chars_per_token` comes from Settings so the ratio can be retuned per
    model family without a code change.
    """
    chars = 0
    for m in messages:
        chars += len(m.content or "")
        for tc in m.tool_calls:
            chars += len(tc.name) + sum(len(str(v)) for v in tc.arguments.values())
    return chars // max(1, chars_per_token)


def _split_recent(
    messages: list[Message], keep_recent: int
) -> tuple[list[Message], list[Message]]:
    """Split messages into (older, recent).

    Walks backwards, skipping the SYSTEM prompt at index 0 (never summarized).
    Keeps the last `keep_recent` non-system messages verbatim; everything else
    is "older" and eligible for summarization.
    """
    recent: list[Message] = []
    older: list[Message] = []
    seen = 0
    i = len(messages) - 1
    while i >= 0:
        m = messages[i]
        if m.role == MessageRole.SYSTEM:
            i -= 1
            continue
        if seen < keep_recent:
            recent.insert(0, m)
            seen += 1
        else:
            older.insert(0, m)
        i -= 1
    return older, recent


def _summarize(messages: list[Message]) -> str:
    """Naive extractive summary — MVP-grade.

    Real implementations would call an LLM with a "summarize this exchange"
    prompt. We don't, to keep tests deterministic and avoid recursion.
    """
    if not messages:
        return "(no prior context)"
    bullets = []
    for m in messages:
        if m.role == MessageRole.USER:
            bullets.append(f"- user: {m.content[:120]}")
        elif m.role == MessageRole.ASSISTANT:
            if m.tool_calls:
                bullets.append(
                    f"- assistant called: {', '.join(tc.name for tc in m.tool_calls)}"
                )
            elif m.content:
                bullets.append(f"- assistant: {m.content[:120]}")
    return "\n".join(bullets[:30])  # hard cap to avoid summary explosion


def _maybe_compress(
    messages: list[Message],
    trigger_tokens: int,
    keep_recent: int,
    chars_per_token: int,
) -> tuple[list[Message], bool]:
    """Apply the sync-summary strategy from Agent中台.md.

    Returns (new_messages, did_compress). The new list starts with the original
    SYSTEM prompt (preserved verbatim), then a synthetic SYSTEM message with
    the summary, followed by the recent raw messages.
    """
    if _estimate_tokens(messages, chars_per_token) < trigger_tokens:
        return messages, False

    # Preserve the original SYSTEM prompt at index 0; never summarize it.
    head: list[Message] = (
        [messages[0]] if messages and messages[0].role == MessageRole.SYSTEM else []
    )
    rest = messages[len(head):]
    older, recent = _split_recent(rest, keep_recent)
    # Never let the boundary orphan a tool result. An assistant message with
    # tool_calls must be immediately followed by the tool messages answering
    # it; a `tool` message whose `tool_call_id` has been summarized away is a
    # hard 400 from every OpenAI-compatible endpoint. _summarize() drops tool
    # messages entirely, so pushing the orphaned replies into `older` puts both
    # halves on the same side of the line.
    while recent and recent[0].role == MessageRole.TOOL:
        older.append(recent.pop(0))
    summary = _summarize(older)
    compressed: list[Message] = [
        *head,
        Message(
            role=MessageRole.SYSTEM,
            content=f"[compressed summary of {len(older)} earlier messages]\n{summary}",
            meta={"compressed": True, "summarized_count": len(older)},
        ),
        *recent,
    ]
    return compressed, True


# ----------------------------------------------------------------------- main


class AgentLoop:
    """One configured Agent, ready to run conversations.

    Holds a ChatModel, a ToolRegistry, a CheckpointStore, a system prompt, and
    a config snapshot. The `run` method is an async generator that emits a
    LoopEvent before each observable step.
    """

    def __init__(
        self,
        *,
        llm: ChatModel,
        tools: ToolRegistry,
        checkpoint: CheckpointStore,
        system_prompt: str,
        settings: Settings,
        config: dict[str, Any] | None = None,
        agent_id: str,
        approvals: ApprovalStore | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.checkpoint = checkpoint
        # Where human approvals live. None means "no store wired" — unit tests
        # and embedded uses — in which case the gate falls back to trusting an
        # approval_id in the request, which is all this platform had before
        # grants existed.
        self.approvals = approvals
        self.system_prompt = system_prompt
        self.settings = settings
        self.config = config or {}
        self.agent_id = agent_id

        # ---- per-agent tool policy ----
        # `tools` narrows what this agent is offered. An empty list means "no
        # restriction" rather than "no tools": agents created without an
        # explicit list, and every config predating this field, keep working
        # instead of silently losing their tools.
        allowed = [str(n) for n in (self.config.get("tools") or [])]
        self._allowed_tools: set[str] | None = set(allowed) or None

        # `sensitive_tools` can only ADD to what the code declares. A tool
        # marked sensitive on the class (WriteFileTool) stays gated even if an
        # operator omits it, because silently dropping an approval gate is not
        # a mistake configuration should be able to make. Use it to gate an
        # otherwise-safe tool, not to un-gate a dangerous one.
        self._sensitive_tools: set[str] = self.tools.sensitive_tools() | {
            str(n) for n in (self.config.get("sensitive_tools") or [])
        }

    def tool_is_allowed(self, name: str) -> bool:
        """May this agent use this tool? Enforced before every execution."""
        return self._allowed_tools is None or name in self._allowed_tools

    def tool_is_sensitive(self, name: str) -> bool:
        """Does calling this tool require prior human approval?"""
        return name in self._sensitive_tools

    def approval_scope_for(self, tc: ToolCall) -> str:
        """What a human approval of this call would cover.

        Delegated to the tool: only it knows what its risk attaches to. A
        write_file approval covers one resolved path; a tool that does not
        narrow the scope is approved as a whole.
        """
        try:
            tool = self.tools.get(tc.name)
        except KeyError:
            return ""
        return tool.approval_scope(tc.arguments)

    async def _requires_approval(
        self, tc: ToolCall, *, user_id: str, approval_id: str | None
    ) -> bool:
        """Does this call still need a human decision?

        Sensitivity is necessary but not sufficient. A live grant covering
        exactly this target means a human already said yes to exactly this
        thing, and re-asking would train people to click approve without
        reading. Grants expire, so this is re-evaluated on every turn.
        """
        if not self.tool_is_sensitive(tc.name):
            return False

        if self.approvals is None:
            # No store: an approval_id in the request is the only signal
            # available. Kept so tests and embedded callers keep working.
            return approval_id is None

        scope = self.approval_scope_for(tc)
        granted = await self.approvals.is_granted(
            agent_id=self.agent_id,
            user_id=user_id,
            tool_name=tc.name,
            scope=scope,
        )
        if granted:
            log.info(
                "approval.reused agent_id=%s tool=%s scope=%s",
                self.agent_id,
                tc.name,
                scope or "(tool-wide)",
            )
        return not granted

    # ------------------------------------------------------------------ public

    async def run(
        self,
        *,
        thread_id: str,
        user_id: str,
        user_message: str | None = None,
        approval_id: str | None = None,
    ) -> AsyncIterator[LoopEvent]:
        """Drive one conversation turn (or resume an interrupted one).

        Args:
            thread_id: stable per-conversation id; also the Checkpoint key.
            user_id: for tool context (audit, sandbox, etc. in later iterations).
            user_message: the new user input. None when resuming from HITL.
            approval_id: when resuming after HITL, the approved request id.
        """
        # A blank user message is not a message. The API contract lets a resume
        # call carry `content: ""`, and honouring that literally used to insert
        # an empty USER turn right after the assistant's still-unanswered
        # tool_calls — which every OpenAI-compatible endpoint rejects with
        # "an assistant message with 'tool_calls' must be followed by tool
        # messages". Normalising here also re-arms the resume-noop
        # short-circuit below, which keys off `user_message is None`.
        if user_message is not None and not user_message.strip():
            user_message = None

        # 1. Resolve starting state.
        snap: CheckpointSnapshot | None = await self.checkpoint.load(thread_id)
        is_resume = snap is not None
        messages: list[Message] = list(snap.messages) if snap else []
        turn = snap.turn if snap else 0
        # Idempotency bookkeeping carried across the loop.
        pending_tools: dict[str, ToolPendingState] = (
            dict(snap.pending_tools) if snap else {}
        )
        done_results: dict[str, str] = dict(snap.done_results) if snap else {}

        if not messages or messages[0].role != MessageRole.SYSTEM:
            messages.insert(
                0, Message(role=MessageRole.SYSTEM, content=self.system_prompt)
            )
        # A user message must never land between an assistant's tool_calls and
        # the tool messages answering them. On a resume the snapshot can still
        # end with an unanswered tool_calls — executing those tools appends
        # their replies below — so the accompanying input is held back until
        # after the replies are in place.
        deferred_user_message: str | None = None
        unresolved_calls = bool(messages) and (
            messages[-1].role == MessageRole.ASSISTANT and bool(messages[-1].tool_calls)
        )
        if user_message is not None:
            if unresolved_calls and is_resume and approval_id is not None:
                deferred_user_message = user_message
            else:
                messages.append(Message(role=MessageRole.USER, content=user_message))

        # 2. Emit start. Include resume marker so the API can refresh UI state,
        # plus the resolved LLM identity — without it, "the agent answered in
        # mock" was indistinguishable from "the agent is broken" when reading
        # the event stream.
        yield LoopEvent(
            type=EventType.START,
            thread_id=thread_id,
            turn=turn,
            data={
                "resumed": is_resume,
                "approval_id": approval_id,
                "agent_id": self.agent_id,
                "model": self.config.get("model") or "(default)",
                "llm": type(self.llm).__name__,
            },
        )
        # NOTE: hitl_resolved is emitted once, by the resume shortcut below,
        # and only when there really was an approval waiting. Emitting it here
        # as well double-fired it on every resume.
        empty_attempts = 0
        # The tool calls this run is asking approval for. Set by whichever path
        # parks the thread and read by the HITL handler below. A local rather
        # than instance state, because one AgentLoop instance serves every
        # concurrent run of its agent.
        hitl_calls: list[ToolCall] = []

        # HITL-resume shortcut: if we're resuming an approved WAITING_APPROVAL
        # thread, the LLM was the one that decided to call those tools in the
        # first place — we don't need to ask it again. Just execute the
        # previously-recorded tool calls and continue.
        if (
            is_resume
            and approval_id is not None
            and snap is not None
            and snap.status == CheckpointStatus.WAITING_APPROVAL
        ):
            # Find the last assistant message that has tool_calls.
            pending_resume_calls: list[ToolCall] = []
            for m in reversed(messages):
                if m.role == MessageRole.ASSISTANT and m.tool_calls:
                    pending_resume_calls = list(m.tool_calls)
                    break
            # Skip any tool calls that already have a DONE result. This matters
            # when the resume is a no-op replay (e.g. caller retries because
            # they lost the SSE response). If everything is already done the
            # snapshot holds both the assistant tool_calls message and its
            # replies, so we fall through to the main loop and let the LLM
            # finalize — re-executing would double-write.
            pending_resume_calls = [
                tc for tc in pending_resume_calls if tc.id not in done_results
            ]

            # Re-check the grants BEFORE announcing that the approval resolved.
            # Presenting an approval_id is not evidence that a human decided
            # anything; only a live grant in the store is. Announcing first and
            # discovering this afterwards would emit hitl_resolved immediately
            # followed by hitl_required, which reads as a glitch rather than as
            # the gate doing its job.
            ungranted = [
                tc
                for tc in pending_resume_calls
                if await self._requires_approval(
                    tc, user_id=user_id, approval_id=approval_id
                )
            ]
            if ungranted:
                # Park directly rather than raising: this shortcut runs before
                # the try/except that normally handles HITLInterrupt, so a
                # raise here would escape `run()` as an unhandled exception
                # instead of a hitl_required event.
                first = ungranted[0]
                yield await self._park_for_approval(
                    interrupt=HITLInterrupt(
                        approval_id=f"appr_{thread_id}_{first.id}",
                        tool_name=first.name,
                        tool_arguments=first.arguments,
                        description=(
                            f"Approval for '{first.name}' on "
                            f"{self.approval_scope_for(first) or 'this tool'} is "
                            f"missing or expired. Approve to continue."
                        ),
                    ),
                    calls=list(pending_resume_calls),
                    messages=messages,
                    thread_id=thread_id,
                    user_id=user_id,
                    turn=turn,
                    pending_tools=pending_tools,
                    done_results=done_results,
                    reason="grant_missing_or_expired",
                )
                return

            if pending_resume_calls:
                yield LoopEvent(
                    type=EventType.HITL_RESOLVED,
                    thread_id=thread_id,
                    turn=turn,
                    data={"approval_id": approval_id, "decision": "approved"},
                )
                # Mark the snapshot as RUNNING so the resume doesn't loop back
                # into WAITING_APPROVAL on a subsequent retry.
                snap.status = CheckpointStatus.RUNNING
                snap.pending_tools = {}
                snap.approval_id = None
                await self.checkpoint.save(snap)
                yield LoopEvent(
                    type=EventType.TOOL_CALL,
                    thread_id=thread_id,
                    turn=turn,
                    data={
                        "calls": [tc.model_dump() for tc in pending_resume_calls],
                        "resumed": True,
                    },
                )
                results = await self._execute_tools(
                    tool_calls=pending_resume_calls,
                    thread_id=thread_id,
                    user_id=user_id,
                    approval_id=approval_id,
                    pending_tools=pending_tools,
                    done_results=done_results,
                )
                for r in results:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=r.content,
                            tool_call_id=r.tool_call_id,
                            meta={
                                "tool_name": r.name,
                                "idempotency_key": r.idempotency_key,
                                "is_error": r.is_error,
                            },
                        )
                    )
                if deferred_user_message is not None:
                    messages.append(
                        Message(
                            role=MessageRole.USER, content=deferred_user_message
                        )
                    )
                    deferred_user_message = None
                yield LoopEvent(
                    type=EventType.TOOL_RESULT,
                    thread_id=thread_id,
                    turn=turn,
                    data={
                        "results": [
                            {
                                "tool_call_id": r.tool_call_id,
                                "name": r.name,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in results
                        ]
                    },
                )
                round_snap = self.checkpoint.new_snapshot(
                    thread_id=thread_id,
                    messages=messages,
                    status=CheckpointStatus.RUNNING,
                    turn=turn + 1,
                    last_config=self.config,
                    user_id=user_id,
                )
                round_snap.pending_tools = pending_tools
                round_snap.done_results = done_results
                await self.checkpoint.save(round_snap)
                turn += 1
                # Fall through to the main loop to let the LLM react to the
                # tool results and produce the final answer.

        try:
            while True:
                if turn >= self.settings.max_turns:
                    # Hard cap per Agent中台.md "单任务最大 100 轮".
                    yield LoopEvent(
                        type=EventType.ERROR,
                        thread_id=thread_id,
                        turn=turn,
                        data={"reason": "max_turns", "max": self.settings.max_turns},
                    )
                    raise LoopBudgetExceeded(self.settings.max_turns)

                # Resume-noop short-circuit: if we're resuming a thread that
                # already finalized in a previous run, just emit finish and
                # return. This protects against caller retries that reissue
                # the resume chat call after they already got their SSE
                # response.
                if (
                    is_resume
                    and user_message is None
                    and messages
                    and messages[-1].role == MessageRole.ASSISTANT
                    and not messages[-1].tool_calls
                ):
                    yield LoopEvent(
                        type=EventType.FINISH,
                        thread_id=thread_id,
                        turn=turn,
                        data={"reason": "already_finalized"},
                    )
                    return

                # ----- before_model -----
                messages, did_compress = _maybe_compress(
                    messages,
                    trigger_tokens=self.settings.compress_trigger_tokens,
                    keep_recent=self.settings.compress_keep_recent_turns,
                    chars_per_token=self.settings.token_estimate_chars_per_token,
                )
                if did_compress:
                    yield LoopEvent(
                        type=EventType.COMPRESSED,
                        thread_id=thread_id,
                        turn=turn,
                        data={
                            "trigger_tokens": self.settings.compress_trigger_tokens
                        },
                    )

                # ----- llm_call -----
                # Cheap invariant check: an unanswered tool call anywhere in
                # the history makes the whole request invalid, so repair before
                # serialising rather than letting the provider 400.
                self._seal_dangling_tool_calls(messages, thread_id)
                llm_messages = [m.to_openai_dict() for m in messages]
                tool_schemas = self.tools.schemas_for_llm(self._allowed_tools)
                response = await self._invoke_llm_with_retries(
                    llm_messages, tool_schemas
                )

                if not response.content and not response.tool_calls:
                    empty_attempts += 1
                    if empty_attempts >= self.settings.empty_response_max_retries:
                        yield LoopEvent(
                            type=EventType.ERROR,
                            thread_id=thread_id,
                            turn=turn,
                            data={
                                "reason": "empty_response",
                                "attempts": empty_attempts,
                            },
                        )
                        raise LoopEmptyResponse(empty_attempts)
                    # silent retry: don't append anything, just loop again
                    continue
                empty_attempts = 0

                # Append the assistant message to history.
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                if response.content:
                    yield LoopEvent(
                        type=EventType.ASSISTANT,
                        thread_id=thread_id,
                        turn=turn,
                        data={"content": response.content},
                    )

                # ----- terminal: no tool calls -> finish -----
                if not response.tool_calls:
                    finished_snap = self.checkpoint.new_snapshot(
                        thread_id=thread_id,
                        messages=messages,
                        status=CheckpointStatus.FINISHED,
                        turn=turn + 1,
                        last_config=self.config,
                        user_id=user_id,
                    )
                    finished_snap.pending_tools = pending_tools
                    finished_snap.done_results = done_results
                    await self.checkpoint.save(finished_snap)
                    yield LoopEvent(
                        type=EventType.FINISH,
                        thread_id=thread_id,
                        turn=turn,
                        data={"reason": "done"},
                    )
                    return

                # ----- after_model: sensitive-tool detection -----
                # A tool this agent may not use is never worth an approval
                # prompt: the answer is "no" regardless of what the human says,
                # so asking would be noise. _execute_tools refuses it instead.
                hitl_target: ToolCall | None = None
                for tc in response.tool_calls:
                    if not self.tool_is_allowed(tc.name):
                        continue
                    if await self._requires_approval(
                        tc, user_id=user_id, approval_id=approval_id
                    ):
                        hitl_target = tc
                        break
                if hitl_target is not None:
                    # Everything in this response is parked, not just the
                    # sensitive one, so the resume replays the batch coherently.
                    hitl_calls = list(response.tool_calls)
                    # Raise HITLInterrupt. The outer try/except catches it,
                    # persists Checkpoint with WAITING_APPROVAL, and exits.
                    raise HITLInterrupt(
                        approval_id=f"appr_{thread_id}_{hitl_target.id}",
                        tool_name=hitl_target.name,
                        tool_arguments=hitl_target.arguments,
                        description=(
                            f"Agent wants to call sensitive tool "
                            f"'{hitl_target.name}' with arguments "
                            f"{hitl_target.arguments}. Approve to continue."
                        ),
                    )

                # ----- tool_executor -----
                yield LoopEvent(
                    type=EventType.TOOL_CALL,
                    thread_id=thread_id,
                    turn=turn,
                    data={"calls": [tc.model_dump() for tc in response.tool_calls]},
                )

                results = await self._execute_tools(
                    tool_calls=response.tool_calls,
                    thread_id=thread_id,
                    user_id=user_id,
                    approval_id=approval_id,
                    pending_tools=pending_tools,
                    done_results=done_results,
                )
                # Append tool results to history.
                for r in results:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=r.content,
                            tool_call_id=r.tool_call_id,
                            meta={
                                "tool_name": r.name,
                                "idempotency_key": r.idempotency_key,
                                "is_error": r.is_error,
                            },
                        )
                    )
                yield LoopEvent(
                    type=EventType.TOOL_RESULT,
                    thread_id=thread_id,
                    turn=turn,
                    data={
                        "results": [
                            {
                                "tool_call_id": r.tool_call_id,
                                "name": r.name,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in results
                        ]
                    },
                )

                # ----- checkpoint after tool round -----
                round_snap = self.checkpoint.new_snapshot(
                    thread_id=thread_id,
                    messages=messages,
                    status=CheckpointStatus.RUNNING,
                    turn=turn + 1,
                    last_config=self.config,
                    user_id=user_id,
                )
                round_snap.pending_tools = pending_tools
                round_snap.done_results = done_results
                await self.checkpoint.save(round_snap)

                turn += 1
                # Loop again.
        except (LoopBudgetExceeded, LoopEmptyResponse):
            # These two already emitted an ERROR event before raising.
            # Re-raise so callers/tests can observe the failure mode
            # explicitly rather than treating it as a generic error.
            raise
        except HITLInterrupt as interrupt:
            # `hitl_calls` is set by whichever path raised. It used to read the
            # response captured here, which only the first-pass path ever
            # populated — the resume path re-parking tripped an assert.
            yield await self._park_for_approval(
                interrupt=interrupt,
                calls=hitl_calls,
                messages=messages,
                thread_id=thread_id,
                user_id=user_id,
                turn=turn,
                pending_tools=pending_tools,
                done_results=done_results,
            )
            return
        except Exception as e:
            # Any other failure — an LLM returning 401 / 429 / timeout, a
            # transport error, a programming bug in a middleware — must NOT
            # escape as an unhandled exception. Doing so tears down the SSE
            # response and the caller sees a bare connection close instead
            # of a diagnosable message.
            #
            # Emit an `error` event and exit cleanly so the frontend can
            # render the reason. Critically, we do this *after* the stream
            # has started, which is the only way the client can see it.
            log.exception(
                "loop.unhandled_error thread_id=%s agent_id=%s",
                thread_id,
                self.agent_id,
            )
            yield LoopEvent(
                type=EventType.ERROR,
                thread_id=thread_id,
                turn=turn,
                data={
                    "reason": "internal_error",
                    "error_type": type(e).__name__,
                    "message": str(e),
                },
            )
            return

    # ----------------------------------------------------------------- helpers

    async def _invoke_llm_with_retries(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> Any:
        """Call the LLM, retrying only failures that could plausibly clear.

        Providers signal this via `LLMError.retryable`. Retrying a 400
        (unknown model name, malformed request) just burns the backoff
        budget and fails identically — the operator waits 3.5s to learn
        something the first response already said.

        Empty responses are handled by the caller (LoopEmptyResponse), not
        here.
        """
        last_err: Exception | None = None
        for attempt in range(self.settings.llm_max_retries):
            try:
                return await self.llm.ainvoke(messages, tool_schemas)
            except Exception as e:
                last_err = e
                # An exception without the attribute is assumed retryable:
                # we would rather over-retry an unknown error than turn a
                # transient blip into a hard failure.
                if getattr(e, "retryable", True) is False:
                    log.warning(
                        "llm.no_retry attempt=%d error=%s", attempt + 1, e
                    )
                    raise
                delay = self.settings.llm_retry_base_delay * (2 ** attempt)
                log.warning(
                    "llm.retry attempt=%d delay=%.2f error=%s",
                    attempt + 1,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
        assert last_err is not None
        raise last_err

    def _seal_dangling_tool_calls(
        self, messages: list[Message], thread_id: str
    ) -> None:
        """Answer any tool call the history left hanging.

        The provider rule is absolute: an assistant message carrying tool_calls
        must be followed by one tool message per tool_call_id. Several paths can
        strand a call — a rejected approval, a new message sent to a thread that
        is still parked, a history edited by hand — and each one used to surface
        much later as an opaque 400 from the model, naming a message index and
        nothing else.

        Sealing here means the invariant is enforced where the outbound payload
        is built, instead of being re-asserted by every caller. It logs a
        warning rather than staying quiet: reaching this means some other layer
        left the history inconsistent, which is worth knowing about.

        Note this REBUILDS the list rather than appending the missing replies.
        Appending only works when the dangling call happens to be last;
        anything that arrived afterwards — a user turn, for instance — would
        still sit between the call and its reply, and the provider would reject
        the request just the same.
        """
        repaired, notes = repair_tool_call_ordering(messages)
        if not notes:
            return
        log.warning(
            "loop.repaired_message_history thread_id=%s repairs=%s",
            thread_id,
            notes,
        )
        messages[:] = repaired

    async def _park_for_approval(
        self,
        *,
        interrupt: HITLInterrupt,
        calls: list[ToolCall],
        messages: list[Message],
        thread_id: str,
        user_id: str,
        turn: int,
        pending_tools: dict[str, ToolPendingState],
        done_results: dict[str, str],
        reason: str | None = None,
    ) -> LoopEvent:
        """Persist a WAITING_APPROVAL snapshot and build the hitl_required event.

        Shared by both places that park a thread: the first-pass gate (which
        raises HITLInterrupt and is handled below) and the resume path. The
        resume shortcut sits *outside* that try block — it runs before the main
        loop starts — so it cannot simply raise; it has to park directly.
        """
        for tc in calls:
            pending_tools.setdefault(tc.id, ToolPendingState.PENDING)
        waiting = self.checkpoint.new_snapshot(
            thread_id=thread_id,
            messages=messages,
            status=CheckpointStatus.WAITING_APPROVAL,
            turn=turn + 1,
            last_config=self.config,
            user_id=user_id,
        )
        waiting.pending_tools = pending_tools
        waiting.done_results = done_results
        # Persist which approval this thread is parked on, so the admin API can
        # list pending approvals with a usable id instead of reconstructing the
        # format, and so /hitl/approve knows exactly what it is granting.
        waiting.approval_id = interrupt.approval_id
        waiting.pending_tool_name = interrupt.tool_name
        waiting.pending_tool_arguments = dict(interrupt.tool_arguments)
        await self.checkpoint.save(waiting)

        data: dict[str, Any] = {
            "approval_id": interrupt.approval_id,
            "tool_name": interrupt.tool_name,
            "tool_arguments": interrupt.tool_arguments,
            "description": interrupt.description,
        }
        if reason is not None:
            data["reason"] = reason
        return LoopEvent(
            type=EventType.HITL_REQUIRED,
            thread_id=thread_id,
            turn=turn,
            data=data,
        )

    async def _execute_tools(
        self,
        *,
        tool_calls: list[ToolCall],
        thread_id: str,
        user_id: str,
        approval_id: str | None,
        pending_tools: dict[str, ToolPendingState],
        done_results: dict[str, str],
    ) -> list[ToolResult]:
        """Run all tool calls in parallel, with idempotency-key bookkeeping."""

        async def one(tc: ToolCall) -> ToolResult:
            # Idempotency: stamp the key onto the ToolCall so downstream sees it.
            if tc.idempotency_key is None:
                tc.idempotency_key = new_idempotency_key(thread_id, tc.id)

            # Allow-list enforcement. The schema list we send the model is an
            # offering, not a gate — a hallucinated or prompt-injected tool name
            # would otherwise reach here and run. Refuse before touching the tool.
            if not self.tool_is_allowed(tc.name):
                return ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=(
                        f"error: tool '{tc.name}' is not enabled for agent "
                        f"'{self.agent_id}'"
                    ),
                    is_error=True,
                    idempotency_key=tc.idempotency_key,
                )

            # On resume from HITL: short-circuit if we already have a DONE result.
            if tc.id in done_results:
                return ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=done_results[tc.id],
                    idempotency_key=tc.idempotency_key,
                )

            try:
                tool = self.tools.get(tc.name)
            except KeyError as e:
                return ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=f"error: {e}",
                    is_error=True,
                    idempotency_key=tc.idempotency_key,
                )

            try:
                output = await tool.run(
                    tc.arguments,
                    ToolContext(
                        thread_id=thread_id,
                        user_id=user_id,
                        approval_id=approval_id,
                    ),
                )
                pending_tools[tc.id] = ToolPendingState.DONE
                done_results[tc.id] = output
            except ToolPermissionDenied as e:
                pending_tools[tc.id] = ToolPendingState.FAILED
                return ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=f"permission denied: {e}",
                    is_error=True,
                    idempotency_key=tc.idempotency_key,
                )
            except Exception as e:
                pending_tools[tc.id] = ToolPendingState.FAILED
                return ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=f"tool error: {e}",
                    is_error=True,
                    idempotency_key=tc.idempotency_key,
                )
            return ToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                content=output,
                idempotency_key=tc.idempotency_key,
            )

        return await asyncio.gather(*(one(tc) for tc in tool_calls))
