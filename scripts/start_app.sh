#!/bin/bash
# Start (or restart) the tdvx app on the rented instance.
# Runs remote-start-multigpu.sh over SSH, then waits for :8000 to respond
# and prints the public URL.
# Usage: ./scripts/start_app.sh

source "$(dirname "$0")/_common.sh"

require_instance
refresh_ssh

info "Launching app on $SSH_HOST:$SSH_PORT..."
ssh_cmd "bash $REMOTE_DIR/scripts/remote-start-multigpu.sh" || die "App launcher failed."

# Resolve the public URL from the vast.ai port mapping.
RAW=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null) || die "Failed to query instance."
PUBLIC_IP=$(echo "$RAW" | jq -r '.public_ipaddr // empty')
PUBLIC_PORT=$(echo "$RAW" | jq -r '.ports["8000/tcp"][0].HostPort // empty')
[[ -n "$PUBLIC_IP" && -n "$PUBLIC_PORT" ]] || die "Could not resolve public URL from vast.ai metadata."
URL="http://$PUBLIC_IP:$PUBLIC_PORT/"

info "Waiting for $URL to respond..."
for i in {1..30}; do
    if curl -fsS -o /dev/null --max-time 5 "$URL"; then
        info "App is live: $URL"
        exit 0
    fi
    echo "  [$i/30] not ready yet..."
    sleep 5
done

die "App did not come up within 150s. Check logs: ./scripts/logs.sh"
