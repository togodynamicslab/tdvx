#!/bin/bash
# Pull human validations + Deepgram references from the server back to local.
# Run this periodically so a future deploy mistake can't lose your judgments.
#
# Files synced:
#   tests/corpus/<lang>/.validations.json
#   tests/corpus/<lang>/.reference/*.json

source "$(dirname "$0")/_common.sh"

refresh_ssh

info "Pulling validations + Deepgram references from $SSH_HOST:$SSH_PORT"
rsync -az \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
    --include='*/' \
    --include='.validations.json' \
    --include='.reference/' \
    --include='.reference/*.json' \
    --exclude='*' \
    "root@$SSH_HOST:$REMOTE_DIR/tests/corpus/" \
    "$REPO_ROOT/tests/corpus/"

info "Done. Local copies updated."
