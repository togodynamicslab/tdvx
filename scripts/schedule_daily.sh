#!/bin/bash
# Daily start/stop dispatcher for the pinned vast.ai instance.
# Called by cron on your laptop. Reads instance ID from .vastai_instance and
# calls vast.ai API directly (no SSH to instance required — stopped instances
# have no SSH).
#
# Usage:
#   scripts/schedule_daily.sh start
#   scripts/schedule_daily.sh stop
#
# Install cron jobs (10am/10pm ET):
#   scripts/schedule_daily.sh install
#   scripts/schedule_daily.sh uninstall
#   scripts/schedule_daily.sh status

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE_FILE="$REPO_ROOT/.vastai_instance"
LOG_DIR="$HOME/.tdvx/logs"
LOG_FILE="$LOG_DIR/schedule.log"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG_FILE"
}

read_key() {
    if [[ -n "${VAST_API_KEY:-}" ]]; then
        echo "$VAST_API_KEY"
        return
    fi
    for f in "$HOME/.config/vastai/vast_api_key" "$HOME/.vastai/vast_api_key"; do
        if [[ -f "$f" ]]; then
            cat "$f"
            return
        fi
    done
    return 1
}

api_call() {
    local state="$1"
    local key instance_id
    key=$(read_key) || { log "ERROR: vast.ai API key not found"; exit 1; }

    [[ -f "$INSTANCE_FILE" ]] || { log "ERROR: $INSTANCE_FILE missing"; exit 1; }
    # shellcheck disable=SC1090
    source "$INSTANCE_FILE"
    instance_id="${INSTANCE_ID:-}"
    [[ -n "$instance_id" ]] || { log "ERROR: INSTANCE_ID missing"; exit 1; }

    log "Requesting state=$state on instance $instance_id"
    local resp
    resp=$(curl -sS -X PUT "https://console.vast.ai/api/v0/instances/$instance_id/" \
        -H "Authorization: Bearer $key" \
        -H "Content-Type: application/json" \
        -d "{\"state\":\"$state\"}" 2>&1) || { log "ERROR: curl failed: $resp"; exit 1; }
    log "Response: $resp"
}

cmd_start() {
    api_call "running"
    log "Daily start dispatched."
}

cmd_stop() {
    api_call "stopped"
    log "Daily stop dispatched."
}

cmd_install() {
    local cron_start="0 10 * * * $REPO_ROOT/scripts/schedule_daily.sh start"
    local cron_stop="0 22 * * * $REPO_ROOT/scripts/schedule_daily.sh stop"
    local marker="# tdvx-daily-schedule"

    # Remove any prior tdvx entries, then append fresh ones.
    # (|| true so grep finding no matches on an empty crontab doesn't trip set -e.)
    local existing
    existing=$( { crontab -l 2>/dev/null || true; } | { grep -v "$marker" || true; } )

    {
        if [[ -n "$existing" ]]; then printf "%s\n" "$existing"; fi
        printf "%s %s\n" "$cron_start" "$marker"
        printf "%s %s\n" "$cron_stop" "$marker"
    } | crontab -

    log "Installed cron: start=10:00 stop=22:00 America/New_York"
    echo ""
    echo "Current crontab:"
    crontab -l | grep -E "(tdvx|schedule_daily)" || true
}

cmd_uninstall() {
    local marker="# tdvx-daily-schedule"
    local existing
    existing=$( { crontab -l 2>/dev/null || true; } | { grep -v "$marker" || true; } )
    if [[ -n "$existing" ]]; then
        printf "%s\n" "$existing" | crontab -
    else
        crontab -r 2>/dev/null || true
    fi
    log "Uninstalled cron entries"
}

cmd_status() {
    echo "Cron entries:"
    crontab -l 2>/dev/null | grep -E "(tdvx|schedule_daily)" || echo "  (none installed — run: $0 install)"
    echo ""
    echo "Local time: $(date)"
    echo "Schedule:   start 10:00 ET, stop 22:00 ET (12h window)"
    echo ""
    echo "Recent log (last 20 lines):"
    tail -n 20 "$LOG_FILE" 2>/dev/null || echo "  (no log yet)"
    echo ""
    if [[ -f "$INSTANCE_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$INSTANCE_FILE"
        echo "Instance: $INSTANCE_ID"
        if command -v vastai >/dev/null; then
            local status
            status=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | jq -r '.actual_status // "unknown"')
            echo "Status:   $status"
        fi
    fi
}

case "${1:-status}" in
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    status)    cmd_status ;;
    *)
        echo "usage: $0 {start|stop|install|uninstall|status}" >&2
        exit 1
        ;;
esac
