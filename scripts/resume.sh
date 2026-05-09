#!/bin/bash
# Resume a stopped instance. GPU billing restarts; disk/files unchanged.
# Checks daily budget before restarting, then starts the app unless --no-app.
# Usage: ./scripts/resume.sh [hours] [--no-app]
#   hours:    planned session length (default: remaining daily budget, capped at 12)
#   --no-app: skip launching the app (box only)

source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/budget.sh"

require_instance

START_APP=1
REQUESTED_HOURS=""
for arg in "$@"; do
    case "$arg" in
        --no-app) START_APP=0 ;;
        *)        REQUESTED_HOURS="$arg" ;;
    esac
done

if [[ -z "$REQUESTED_HOURS" ]]; then
    REQUESTED_HOURS=$(budget_remaining_hours)
    # Cap at MAX_HOURS (default 12) so a fresh day doesn't reserve everything by accident.
    CAP="${MAX_HOURS:-12}"
    REQUESTED_HOURS=$(awk -v r="$REQUESTED_HOURS" -v c="$CAP" 'BEGIN { printf "%.2f", (r < c) ? r : c }')
fi

budget_check "$REQUESTED_HOURS" || exit 1

info "Resuming instance $INSTANCE_ID (planned: ${REQUESTED_HOURS}h)..."
vastai start instance "$INSTANCE_ID" || die "Resume failed."

budget_log "$REQUESTED_HOURS" "$INSTANCE_ID"

info "Waiting for instance to reach 'running'..."
for i in {1..30}; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | jq -r '.actual_status // "unknown"')
    echo "  [$i/30] status=$STATUS"
    [[ "$STATUS" == "running" ]] && break
    sleep 10
done

[[ "$STATUS" == "running" ]] || die "Instance did not resume in 5min. Check 'vastai show instance $INSTANCE_ID'."

info "Resumed. NOTE: self-stop timer does NOT carry across stop/resume."
info "  To re-arm auto-stop, run: ./scripts/arm_stop.sh ${REQUESTED_HOURS}"

if [[ "$START_APP" -eq 1 ]]; then
    # SSH endpoint may need a few seconds after 'running' before it accepts connections.
    sleep 5
    "$(dirname "$0")/start_app.sh"
else
    info "Skipping app start (--no-app). Run ./scripts/start_app.sh when ready."
fi
