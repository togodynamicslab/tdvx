#!/bin/bash
# Rent an 8x RTX 5060 Ti instance on Vast.ai, provision it, save instance ID.
# Usage: ./scripts/rent.sh [max_price_per_hour]

source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/budget.sh"

MAX_DPH="${1:-1.00}"
IMAGE="${TDVX_IMAGE:-ghcr.io/togodynamicslab/tdvx:latest}"
DISK_GB=50
BOOT_TIMEOUT_S=900
MAX_HOURS="${MAX_HOURS:-12}"       # instance self-stops after this many hours (preserves disk)
SKIP_HOSTS="${SKIP_HOSTS:-}"       # space-separated host_ids to exclude
ALLOW_COUNTRIES="${ALLOW_COUNTRIES:-US CA GB DE FR NL SG JP}"  # ISO codes; set to empty to allow all

budget_check "$MAX_HOURS" || exit 1

# GHCR private-registry login for Vast to pull the image.
# Set GHCR_USER and GHCR_TOKEN (a PAT with read:packages) in your shell or .env.
GHCR_USER="${GHCR_USER:-}"
GHCR_TOKEN="${GHCR_TOKEN:-}"
[[ -n "$GHCR_USER" && -n "$GHCR_TOKEN" ]] || die "GHCR_USER and GHCR_TOKEN must be set (PAT with read:packages) to pull the private image."

if [[ -f "$INSTANCE_FILE" ]]; then
    source "$INSTANCE_FILE"
    die "Instance $INSTANCE_ID already exists. Run ./scripts/destroy.sh first or delete $INSTANCE_FILE."
fi

command -v jq >/dev/null || die "jq required. Install: sudo apt install jq"
command -v vastai >/dev/null || die "vastai CLI required. Install: pip install vastai"

# Preferred-host lock: if .vastai_preferred_host exists, only rent from that host.
# Clear the file or pass ALLOW_ANY_HOST=1 to override (e.g. if the host is offline).
PREFERRED_HOST=""
PREFERRED_FILE="$REPO_ROOT/.vastai_preferred_host"
if [[ -f "$PREFERRED_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$PREFERRED_FILE"
    PREFERRED_HOST="${HOST_ID:-}"
fi

info "Searching for 8x RTX 5060 Ti offers under \$${MAX_DPH}/hr..."
OFFERS=$(vastai search offers \
    "gpu_name=RTX_5060_Ti num_gpus=8 gpu_ram>=16 verified=True rentable=True dph<=${MAX_DPH}" \
    -o 'dph' --raw)

OFFER_COUNT=$(echo "$OFFERS" | jq 'length')
[[ "$OFFER_COUNT" -gt 0 ]] || die "No matching offers found. Try raising price or relaxing filters."

# Pin to preferred host if set (takes precedence over country/skip filters).
if [[ -n "$PREFERRED_HOST" && "${ALLOW_ANY_HOST:-0}" != "1" ]]; then
    OFFERS=$(echo "$OFFERS" | jq --argjson h "$PREFERRED_HOST" '[.[] | select(.host_id == $h)]')
    OFFER_COUNT=$(echo "$OFFERS" | jq 'length')
    info "Pinned to preferred host $PREFERRED_HOST: $OFFER_COUNT offers"
    if [[ "$OFFER_COUNT" -eq 0 ]]; then
        die "Preferred host $PREFERRED_HOST has no available offers right now. Wait and retry, or run with ALLOW_ANY_HOST=1 to allow a different host (WARNING: different host = different disk, no persistent data)."
    fi
fi

# Filter by country allow-list (geolocation contains ", <ISO>")
if [[ -n "$ALLOW_COUNTRIES" ]]; then
    ALLOW_REGEX=$(echo "$ALLOW_COUNTRIES" | tr ' ' '|')
    OFFERS=$(echo "$OFFERS" | jq --arg re "$ALLOW_REGEX" '[.[] | select(.geolocation // "" | test(", (" + $re + ")$"))]')
    OFFER_COUNT=$(echo "$OFFERS" | jq 'length')
    info "After country filter [$ALLOW_COUNTRIES]: $OFFER_COUNT offers"
    [[ "$OFFER_COUNT" -gt 0 ]] || die "No offers in allowed countries. Raise price, relax ALLOW_COUNTRIES, or SKIP."
fi

# Filter out blacklisted hosts (e.g. ones that previously stalled)
if [[ -n "$SKIP_HOSTS" ]]; then
    SKIP_JSON=$(printf '%s\n' $SKIP_HOSTS | jq -R . | jq -s 'map(tonumber)')
    OFFERS=$(echo "$OFFERS" | jq --argjson skip "$SKIP_JSON" '[.[] | select(.host_id as $h | $skip | index($h) | not)]')
    OFFER_COUNT=$(echo "$OFFERS" | jq 'length')
    info "After skipping hosts [$SKIP_HOSTS]: $OFFER_COUNT offers remain"
    [[ "$OFFER_COUNT" -gt 0 ]] || die "All matching hosts are blacklisted. Clear SKIP_HOSTS or raise price."
fi

OFFER_ID=$(echo "$OFFERS" | jq -r '.[0].id')
OFFER_PRICE=$(echo "$OFFERS" | jq -r '.[0].dph_total')
OFFER_HOST=$(echo "$OFFERS" | jq -r '.[0].host_id')
info "Picking cheapest: offer=$OFFER_ID host=$OFFER_HOST price=\$${OFFER_PRICE}/hr"

# Vast.ai API key for self-stop. Read from env or ~/.vastai/vast_api_key.
VAST_API_KEY="${VAST_API_KEY:-}"
for keyfile in "$HOME/.config/vastai/vast_api_key" "$HOME/.vastai/vast_api_key"; do
    if [[ -z "$VAST_API_KEY" && -f "$keyfile" ]]; then
        VAST_API_KEY="$(cat "$keyfile")"
    fi
done
[[ -n "$VAST_API_KEY" ]] || die "VAST_API_KEY not set and no key file at ~/.config/vastai/vast_api_key. Self-stop requires API access."

MAX_SECONDS=$(( MAX_HOURS * 3600 ))

# Onstart:
#  - install rsync (needed by push.sh)
#  - schedule a self-stop after MAX_SECONDS (calls vast.ai API from inside the box)
#  - self-stop pauses GPU billing but preserves disk so we can resume later
ONSTART=$(cat <<EOF
apt-get update && apt-get install -y --no-install-recommends rsync curl at
service atd start || true
mkdir -p /workspace/tdvx

cat > /workspace/self_stop.sh <<'STOP'
#!/bin/bash
# Self-stop: pause this instance via vast.ai API. Disk persists.
# Triggered by 'at' timer scheduled at container start.
INSTANCE_ID="\${CONTAINER_ID:-\$VAST_CONTAINERLABEL}"
if [[ -z "\$INSTANCE_ID" ]]; then
    INSTANCE_ID=\$(cat /etc/vastai_instance_id 2>/dev/null || echo "")
fi
echo "[\$(date -u +%FT%TZ)] self-stop firing for instance=\$INSTANCE_ID" >> /workspace/self_stop.log
curl -sS -X PUT "https://console.vast.ai/api/v0/instances/\$INSTANCE_ID/" \\
    -H "Authorization: Bearer \$VAST_API_KEY_SELFSTOP" \\
    -H "Content-Type: application/json" \\
    -d '{"state":"stopped"}' >> /workspace/self_stop.log 2>&1
STOP
chmod +x /workspace/self_stop.sh

# Persist API key + instance id for the scheduled job
echo "export VAST_API_KEY_SELFSTOP='$VAST_API_KEY'" > /workspace/.self_stop_env
echo "\$CONTAINER_ID" > /etc/vastai_instance_id 2>/dev/null || true

echo "[\$(date -u +%FT%TZ)] scheduling self-stop in ${MAX_HOURS}h (${MAX_SECONDS}s)" > /workspace/self_stop.log
echo "source /workspace/.self_stop_env && /workspace/self_stop.sh" | at now + $MAX_HOURS hours 2>> /workspace/self_stop.log

touch /workspace/READY
EOF
)

info "Creating instance (private GHCR image: $IMAGE)..."
CREATE_OUT=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --login "-u $GHCR_USER -p $GHCR_TOKEN https://ghcr.io" \
    --disk "$DISK_GB" \
    --ssh --direct \
    --onstart-cmd "$ONSTART" \
    --env '-p 8000:8000' \
    --label "tdvx-dev" \
    --raw 2>&1)

# Vast CLI returns plain-text errors on failure, JSON on success.
if ! echo "$CREATE_OUT" | jq -e . >/dev/null 2>&1; then
    die "vastai create failed: $CREATE_OUT"
fi

INSTANCE_ID=$(echo "$CREATE_OUT" | jq -r '.new_contract // .id // empty')
if [[ -z "$INSTANCE_ID" ]]; then
    ERR=$(echo "$CREATE_OUT" | jq -r '.error // .detail // .msg // empty')
    die "vastai create returned no instance ID. Error: ${ERR:-unknown}. Full response: $CREATE_OUT"
fi

cat > "$INSTANCE_FILE" <<EOF
INSTANCE_ID=$INSTANCE_ID
OFFER_ID=$OFFER_ID
PRICE_PER_HOUR=$OFFER_PRICE
CREATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HOST_ID=$OFFER_HOST
MAX_HOURS=$MAX_HOURS
EOF

budget_log "$MAX_HOURS" "$INSTANCE_ID"

info "Instance $INSTANCE_ID created at \$${OFFER_PRICE}/hr. Saved to $INSTANCE_FILE."
info "Waiting for instance to boot (timeout ${BOOT_TIMEOUT_S}s, auto-destroy on stall)..."

START_TS=$(date +%s)
STATUS="unknown"
while (( $(date +%s) - START_TS < BOOT_TIMEOUT_S )); do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | jq -r '.actual_status // "unknown"')
    ELAPSED=$(( $(date +%s) - START_TS ))
    echo "  [${ELAPSED}s/${BOOT_TIMEOUT_S}s] status=$STATUS"
    if [[ "$STATUS" == "running" ]]; then
        break
    fi
    sleep 15
done

if [[ "$STATUS" != "running" ]]; then
    ELAPSED=$(( $(date +%s) - START_TS ))
    COST=$(awk "BEGIN { printf \"%.3f\", $OFFER_PRICE * $ELAPSED / 3600 }")
    info "Instance did not reach 'running' in ${BOOT_TIMEOUT_S}s (cost so far: \$${COST})."
    info "Auto-destroying instance $INSTANCE_ID on host $OFFER_HOST..."
    yes | vastai destroy instance "$INSTANCE_ID" || info "Destroy failed (may already be gone)."
    rm -f "$INSTANCE_FILE"
    info "To retry skipping this host: SKIP_HOSTS=$OFFER_HOST ./scripts/rent.sh ${MAX_DPH}"
    exit 1
fi

info "Instance running. Waiting for onstart script (installing rsync/ffmpeg)..."
for i in {1..30}; do
    if ssh_cmd "test -f /workspace/READY" 2>/dev/null; then
        info "Onstart complete."
        break
    fi
    echo "  [$i/30] waiting..."
    sleep 10
done

info ""
info "Ready. Auto-stop in ${MAX_HOURS}h (disk preserved, GPU billing paused)."
info "Next steps:"
info "  ./scripts/push.sh         # push code + start server"
info "  ./scripts/logs.sh         # tail server logs"
info "  ./scripts/ssh.sh          # ssh into the box"
info "  ./scripts/stop.sh         # stop early (keeps disk, pauses billing)"
info "  ./scripts/resume.sh       # resume a stopped instance"
info "  ./scripts/destroy.sh      # fully delete (wipes disk)"
