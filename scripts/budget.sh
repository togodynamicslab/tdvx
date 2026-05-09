#!/bin/bash
# Daily rental-hour budget guard for vast.ai instances.
# Sourced by rent.sh; can also be invoked directly to inspect today's usage.
#
# Model:
#   - Each rent logs a planned duration (MAX_HOURS) to a per-day file.
#   - Before renting, sum today's planned hours; refuse if >= DAILY_BUDGET_HOURS.
#   - Planned-hours model is intentionally conservative: if you destroy early,
#     you can clear today's log with `scripts/budget.sh reset` to reclaim it.
#
# Usage:
#   source scripts/budget.sh          # for rent.sh
#   scripts/budget.sh status          # show today's usage
#   scripts/budget.sh reset           # clear today's log
#   scripts/budget.sh log <hours>     # append entry (called by rent.sh)

DAILY_BUDGET_HOURS="${DAILY_BUDGET_HOURS:-12}"
BUDGET_DIR="${BUDGET_DIR:-$HOME/.tdvx/budget}"
TODAY="$(date -u +%Y-%m-%d)"
BUDGET_FILE="$BUDGET_DIR/$TODAY.log"

mkdir -p "$BUDGET_DIR"
touch "$BUDGET_FILE"

budget_used_hours() {
    awk '{ sum += $2 } END { printf "%.2f", sum + 0 }' "$BUDGET_FILE"
}

budget_remaining_hours() {
    local used
    used=$(budget_used_hours)
    awk -v b="$DAILY_BUDGET_HOURS" -v u="$used" 'BEGIN { printf "%.2f", b - u }'
}

budget_check() {
    local needed_h="${1:-$DAILY_BUDGET_HOURS}"
    local used remaining
    used=$(budget_used_hours)
    remaining=$(budget_remaining_hours)
    local over
    over=$(awk -v n="$needed_h" -v r="$remaining" 'BEGIN { print (n > r) ? 1 : 0 }')
    if [[ "$over" == "1" ]]; then
        echo "ERROR: Budget exceeded. Today used=${used}h, need=${needed_h}h, remaining=${remaining}h, daily cap=${DAILY_BUDGET_HOURS}h." >&2
        echo "       Override with DAILY_BUDGET_HOURS=N, or 'scripts/budget.sh reset' if you destroyed early." >&2
        return 1
    fi
    echo "Budget OK: used=${used}h, reserving=${needed_h}h, remaining after=${remaining}h (cap=${DAILY_BUDGET_HOURS}h)."
    return 0
}

budget_log() {
    local hours="$1"
    local instance_id="${2:-unknown}"
    printf "%s %s %s\n" "$(date -u +%H:%M:%SZ)" "$hours" "$instance_id" >> "$BUDGET_FILE"
}

# CLI mode
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-status}" in
        status)
            echo "Date: $TODAY (UTC)"
            echo "Cap:  ${DAILY_BUDGET_HOURS}h"
            echo "Used: $(budget_used_hours)h"
            echo "Left: $(budget_remaining_hours)h"
            if [[ -s "$BUDGET_FILE" ]]; then
                echo "Entries:"
                cat "$BUDGET_FILE"
            fi
            ;;
        reset)
            : > "$BUDGET_FILE"
            echo "Cleared $BUDGET_FILE"
            ;;
        log)
            [[ -n "${2:-}" ]] || { echo "usage: $0 log <hours> [instance_id]" >&2; exit 1; }
            budget_log "$2" "${3:-manual}"
            echo "Logged ${2}h."
            ;;
        check)
            budget_check "${2:-$DAILY_BUDGET_HOURS}"
            ;;
        *)
            echo "usage: $0 {status|reset|log <hours> [id]|check [hours]}" >&2
            exit 1
            ;;
    esac
fi
