#!/bin/bash
# Stop the rented instance (pauses GPU billing, preserves disk).
# Files, models, checkpoints all persist. Resume with ./scripts/resume.sh.
# Usage: ./scripts/stop.sh [-y]

source "$(dirname "$0")/_common.sh"

require_instance

if [[ "${1:-}" != "-y" ]]; then
    read -r -p "Stop instance $INSTANCE_ID? Disk persists (~\$0.10/GB/day). [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || { info "Aborted."; exit 1; }
fi

info "Stopping instance $INSTANCE_ID..."
vastai stop instance "$INSTANCE_ID" || die "Stop failed."
info "Stopped. GPU billing paused, disk preserved."
info "Resume with: ./scripts/resume.sh"
info "Fully delete with: ./scripts/destroy.sh"
