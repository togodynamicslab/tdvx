#!/bin/bash
# Open an interactive SSH session to the rented instance.
# Usage: ./scripts/ssh.sh [command...]

source "$(dirname "$0")/_common.sh"

refresh_ssh

if [[ $# -eq 0 ]]; then
    exec ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -p "$SSH_PORT" "root@$SSH_HOST"
else
    ssh_cmd "$@"
fi
