#!/bin/bash
# Tail the uvicorn log on the rented instance. Ctrl-C to stop.
# Usage: ./scripts/logs.sh [-n <lines>]

source "$(dirname "$0")/_common.sh"

LINES="${1:-100}"
if [[ "${1:-}" == "-n" ]]; then LINES="${2:-100}"; fi

ssh_cmd "tail -n $LINES -f /workspace/tdvx.log"
