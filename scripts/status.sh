#!/bin/bash
# Show instance status, SSH endpoint, public URL, uptime, and cost so far.
# Usage: ./scripts/status.sh

source "$(dirname "$0")/_common.sh"

require_instance

RAW=$(vastai show instance "$INSTANCE_ID" --raw) || die "Failed to query instance $INSTANCE_ID"

STATUS=$(echo "$RAW" | jq -r '.actual_status // "unknown"')
SSH_HOST=$(echo "$RAW" | jq -r '.ssh_host // "n/a"')
SSH_PORT=$(echo "$RAW" | jq -r '.ssh_port // "n/a"')
PUBLIC_IP=$(echo "$RAW" | jq -r '.public_ipaddr // "n/a"')
MAPPED_PORT=$(echo "$RAW" | jq -r '.ports["8000/tcp"][0].HostPort // "n/a"')
DPH=$(echo "$RAW" | jq -r '.dph_total // 0')
START_TS=$(echo "$RAW" | jq -r '.start_date // 0')

NOW=$(date +%s)
if [[ "$START_TS" != "0" && "$START_TS" != "null" ]]; then
    ELAPSED_S=$((NOW - ${START_TS%.*}))
    ELAPSED_H=$(awk "BEGIN { printf \"%.2f\", $ELAPSED_S / 3600 }")
    COST=$(awk "BEGIN { printf \"%.3f\", $DPH * $ELAPSED_S / 3600 }")
else
    ELAPSED_H="n/a"
    COST="n/a"
fi

cat <<EOF
Instance:       $INSTANCE_ID
Status:         $STATUS
SSH:            ssh -p $SSH_PORT root@$SSH_HOST
Public URL:     http://$PUBLIC_IP:$MAPPED_PORT
Rate:           \$${DPH}/hr
Uptime:         ${ELAPSED_H}h
Cost so far:    \$${COST}
EOF
