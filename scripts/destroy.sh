#!/bin/bash
# Fully delete the rented instance. WIPES DISK — checkpoints, datasets, models all lost.
# For the normal 12h-off workflow use ./scripts/stop.sh instead (pauses billing, keeps disk).
# Usage: ./scripts/destroy.sh [-y]

source "$(dirname "$0")/_common.sh"

require_instance

PREFERRED_FILE="$REPO_ROOT/.vastai_preferred_host"
PINNED_HOST=""
if [[ -f "$PREFERRED_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$PREFERRED_FILE"
    PINNED_HOST="${HOST_ID:-}"
fi

if [[ "${1:-}" != "-y" ]]; then
    echo ""
    echo "================================================================"
    echo "  WARNING: This DESTROYS instance $INSTANCE_ID."
    echo "  - Disk will be wiped (models, datasets, checkpoints lost)"
    echo "  - Billing stops fully"
    if [[ -n "$PINNED_HOST" ]]; then
        echo "  - You'll lose your pinned host $PINNED_HOST's persistent state"
        echo "  - To only PAUSE (keep disk): ./scripts/stop.sh"
    fi
    echo "================================================================"
    read -r -p "Type 'destroy' to confirm: " ans
    [[ "$ans" == "destroy" ]] || { info "Aborted."; exit 1; }
fi

info "Destroying instance $INSTANCE_ID..."
yes | vastai destroy instance "$INSTANCE_ID" || info "Destroy command failed (instance may already be gone)."

rm -f "$INSTANCE_FILE"
info "Done. $INSTANCE_FILE removed. Preferred-host config kept at $PREFERRED_FILE."
