#!/bin/bash
# Shared helpers for vast.ai deploy scripts.
# Source this from other scripts: source "$(dirname "$0")/_common.sh"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE_FILE="$REPO_ROOT/.vastai_instance"
REMOTE_DIR="/workspace/tdvx"

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo ">> $*"; }

require_instance() {
    [[ -f "$INSTANCE_FILE" ]] || die "No instance. Run ./scripts/rent.sh first."
    # shellcheck disable=SC1090
    source "$INSTANCE_FILE"
    [[ -n "${INSTANCE_ID:-}" ]] || die "INSTANCE_ID missing from $INSTANCE_FILE"
}

# Fetch fresh SSH host:port from vast.ai (they can change after restarts).
# Prefers direct endpoint (public_ipaddr + mapped 22/tcp port) over the
# ssh_host:ssh_port proxy, which doesn't work for --direct instances.
refresh_ssh() {
    require_instance
    local raw
    raw=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null) || die "Failed to query instance $INSTANCE_ID"
    STATUS=$(echo "$raw" | jq -r '.actual_status // .intended_status // "unknown"')

    # Prefer direct endpoint
    local direct_ip direct_port
    direct_ip=$(echo "$raw" | jq -r '.public_ipaddr // empty')
    direct_port=$(echo "$raw" | jq -r '.ports["22/tcp"][0].HostPort // empty')
    if [[ -n "$direct_ip" && -n "$direct_port" ]]; then
        SSH_HOST="$direct_ip"
        SSH_PORT="$direct_port"
        return
    fi

    # Fallback to proxy endpoint
    SSH_HOST=$(echo "$raw" | jq -r '.ssh_host // empty')
    SSH_PORT=$(echo "$raw" | jq -r '.ssh_port // empty')
    [[ -n "$SSH_HOST" && -n "$SSH_PORT" ]] || die "SSH endpoint not available yet (status=$STATUS). Wait a bit and retry."
}

ssh_cmd() {
    refresh_ssh
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -p "$SSH_PORT" "root@$SSH_HOST" "$@"
}
