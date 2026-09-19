"""Static lint: stdlib loggers must not be called with arbitrary kwargs.

Whatever the merits of structlog, the codebase currently uses
`logging.getLogger()` — whose `_log()` accepts only `exc_info`,
`stack_info`, `stacklevel`, and `extra`. A call written in structlog
style, e.g.

    log.warning("llm.retry", attempt=n, delay=d, error=str(e))

raises `TypeError: Logger._log() got an unexpected keyword argument
'attempt'
**at runtime**, inside the very error path it was meant to report. That
bug class has bitten this project three separate times (seed_defaults,
checkpoint load, llm retry), each time masking the real failure. This
module makes it impossible to reintroduce silently.

If structlog is adopted properly later, delete this test and replace the
stdlib loggers with `structlog.get_logger()`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent_platform.config.settings import project_root

# Resolved the same way the application does, so relocating the test suite
# cannot silently repoint it (it did, when the suite moved under resource/).
_repo_root = project_root()
assert _repo_root is not None, "tests must run from a source checkout"
SRC_ROOT = _repo_root / "src" / "agent_platform"

# The only keyword arguments stdlib logging actually accepts on the
# convenience methods.
ALLOWED_LOGGING_KWARGS = {"exc_info", "stack_info", "stacklevel", "extra"}

LOG_METHODS = {
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "exception",
    "critical",
}


def _iter_python_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _find_kwarg_log_calls(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return (lineno, method_name, offending_kwargs) for violations."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match `log.<method>(...)`.
        if not isinstance(func, ast.Attribute) or func.attr not in LOG_METHODS:
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "log":
            continue
        offending = [
            kw.arg
            for kw in node.keywords
            if kw.arg is not None and kw.arg not in ALLOWED_LOGGING_KWARGS
        ]
        if offending:
            out.append((node.lineno, func.attr, offending))
    return out


def test_python_files_discovered() -> None:
    """Guard against the scan silently matching nothing."""
    files = _iter_python_files()
    assert len(files) > 10, "only found " + str(len(files)) + " source files"


@pytest.mark.parametrize("path", _iter_python_files(), ids=lambda p: p.name)
def test_no_stdlib_log_calls_with_arbitrary_kwargs(path: Path) -> None:
    violations = _find_kwarg_log_calls(path)
    if violations:
        lines = ", ".join(
            "line " + str(ln) + ": log." + m + "(..., " + ", ".join(kw) + ")"
            for ln, m, kw in violations
        )
        pytest.fail(
            str(path)
            + " uses structlog-style kwargs on a stdlib logger — this raises"
            + " TypeError at runtime. Use %-formatting instead. Violations: "
            + lines
        )
