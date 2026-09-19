#!/usr/bin/env bash
#
# End-to-end HITL walkthrough against a running server.
#
# Covers the arc that matters and is easy to get wrong:
#   ordinary turn -> sensitive tool -> SUSPEND -> approve -> resume -> finish
# and the reject branch, which must end the conversation without running the tool.
#
# It asserts the two things that make HITL HITL:
#   * the tool does NOT run before approval
#   * the resume picks up the same thread instead of restarting it
#
# Usage:
#   .venv/bin/python -m agent_platform.cli serve --host 127.0.0.1 --port 8000 &
#   scripts/hitl_walkthrough.sh                 # defaults to :8000, agent demo
#   scripts/hitl_walkthrough.sh http://127.0.0.1:8011 demo
#
# --noproxy '*' is deliberate: a proxy in http_proxy will otherwise swallow
# requests to localhost on some setups.

set -uo pipefail

BASE="${1:-http://127.0.0.1:8000}"
AGENT="${2:-demo}"
CURL=(curl -sS --noproxy '*')

FAILURES=0
say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '   \033[31mFAIL\033[0m %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

# Pull the first non-null approval_id out of an SSE transcript. The `start`
# event also carries the field, set to null, so matching on the appr_ prefix
# is what makes this reliable.
approval_of() { printf '%s' "$1" | grep -o '"approval_id": *"appr_[^"]*"' | head -1 | sed 's/.*"\(appr_[^"]*\)".*/\1/'; }
thread_of()   { printf '%s' "$1" | sed 's/.*"thread_id":"\([^"]*\)".*/\1/'; }

printf 'server : %s\nagent  : %s\n' "$BASE" "$AGENT"

# --------------------------------------------------------------------------
say "1. create a session"
SESSION=$("${CURL[@]}" -X POST "$BASE/v1/sessions" -H 'content-type: application/json' \
  -d "{\"agent_id\":\"$AGENT\",\"user_id\":\"hitl-walkthrough\"}")
SID=$(thread_of "$SESSION")
if [ -z "$SID" ]; then bad "no thread_id: $SESSION"; exit 1; fi
echo "   thread_id = $SID"

# --------------------------------------------------------------------------
say "2. a turn that reaches for the sensitive tool"
CHAT=$("${CURL[@]}" -N -X POST "$BASE/v1/sessions/$SID/chat" -H 'content-type: application/json' \
  -d '{"content":"Use the write_file tool to save the text approved-by-a-human into notes/hitl.txt."}')
printf '%s\n' "$CHAT" | sed 's/^/   /'

APPROVAL=$(approval_of "$CHAT")
if [ -z "$APPROVAL" ]; then
  bad "no approval_id — the model never asked for write_file. Is write_file in the registry?"
  exit 1
fi
echo "   approval_id = $APPROVAL"
printf '%s' "$CHAT" | grep -q 'event: hitl_required' && ok "suspended: hitl_required emitted" \
  || bad "expected a hitl_required event"
printf '%s' "$CHAT" | grep -q 'event: tool_result' \
  && bad "the tool ran BEFORE approval — HITL is not gating anything" \
  || ok "the sensitive tool did not run before approval"

# The Checkpoint must now be parked in waiting_approval.
STATUS=$("${CURL[@]}" "$BASE/admin/api/checkpoints/$SID" | grep -o '"status": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)".*/\1/')
[ "$STATUS" = "waiting_approval" ] && ok "checkpoint status = waiting_approval" \
  || bad "checkpoint status = '$STATUS', expected waiting_approval"

# --------------------------------------------------------------------------
say "3. approve"
"${CURL[@]}" -X POST "$BASE/v1/sessions/$SID/hitl/approve" -H 'content-type: application/json' \
  -d "{\"approval_id\":\"$APPROVAL\"}" | sed 's/^/   /'
echo

# --------------------------------------------------------------------------
say "4. resume — empty content is NOT a user turn"
RESUME=$("${CURL[@]}" -N -X POST "$BASE/v1/sessions/$SID/chat" -H 'content-type: application/json' \
  -d "{\"content\":\"\",\"approval_id\":\"$APPROVAL\"}")
printf '%s\n' "$RESUME" | sed 's/^/   /'

printf '%s' "$RESUME" | grep -q 'event: hitl_resolved' && ok "hitl_resolved emitted" \
  || bad "expected hitl_resolved on resume"
printf '%s' "$RESUME" | grep -q 'event: tool_result' && ok "the tool ran after approval" \
  || bad "the tool still did not run after approval"
printf '%s' "$RESUME" | grep -q 'event: finish' && ok "finished" || bad "resume never finished"

APPROVED_FILE="workspace/notes/hitl.txt"
if [ -f "$APPROVED_FILE" ]; then
  ok "wrote $APPROVED_FILE -> $(cat "$APPROVED_FILE")"
else
  bad "$APPROVED_FILE missing (is the server's CWD the project root?)"
fi

# --------------------------------------------------------------------------
say "5. the reject branch, on a fresh thread"
SESSION2=$("${CURL[@]}" -X POST "$BASE/v1/sessions" -H 'content-type: application/json' \
  -d "{\"agent_id\":\"$AGENT\",\"user_id\":\"hitl-walkthrough\"}")
SID2=$(thread_of "$SESSION2")
CHAT2=$("${CURL[@]}" -N -X POST "$BASE/v1/sessions/$SID2/chat" -H 'content-type: application/json' \
  -d '{"content":"Use the write_file tool to save NOPE into notes/rejected.txt."}')
APPROVAL2=$(approval_of "$CHAT2")
if [ -z "$APPROVAL2" ]; then
  bad "second thread never reached HITL"
else
  "${CURL[@]}" -X POST "$BASE/v1/sessions/$SID2/hitl/reject" -H 'content-type: application/json' \
    -d "{\"approval_id\":\"$APPROVAL2\"}" | sed 's/^/   /'
  echo
  STATUS2=$("${CURL[@]}" "$BASE/admin/api/checkpoints/$SID2" | grep -o '"status": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)".*/\1/')
  [ "$STATUS2" = "finished" ] && ok "rejected thread is finished" \
    || bad "rejected thread status = '$STATUS2', expected finished"
  [ -f workspace/notes/rejected.txt ] \
    && bad "the tool ran despite rejection!" \
    || ok "nothing was written on rejection"
fi

# --------------------------------------------------------------------------
printf '\n'
if [ "$FAILURES" -eq 0 ]; then
  printf '\033[32mAll HITL checks passed.\033[0m\n'
else
  printf '\033[31m%d check(s) failed.\033[0m\n' "$FAILURES"
fi
exit "$FAILURES"
