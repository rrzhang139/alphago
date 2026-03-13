#!/bin/bash
# Poll the experiment queue for new requests.
# Usage: .claude/infra/poll_queue.sh [interval_seconds]
#
# Runs in a loop: git pull, check for *.json in experiments/queue/,
# exit 0 with the filename when found. Exit 1 on interrupt.
#
# The infra agent calls this and blocks until a new request appears.
# This is cheap — no LLM tokens burned while waiting.

INTERVAL="${1:-60}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE_DIR="$REPO_ROOT/experiments/queue"

echo "[poll] Watching $QUEUE_DIR every ${INTERVAL}s..."

while true; do
    # Pull latest
    cd "$REPO_ROOT" && git pull --quiet 2>/dev/null

    # Check for .json files (excluding done/ subdirectory and README)
    FOUND=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.json' -type f 2>/dev/null | head -1)

    if [ -n "$FOUND" ]; then
        echo "[poll] Found: $FOUND"
        echo "$FOUND"
        exit 0
    fi

    sleep "$INTERVAL"
done
