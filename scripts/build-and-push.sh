#!/bin/bash
# Build the tdvx image and push to private GHCR.
# Requires: docker login ghcr.io (one-time), or GHCR_USER + GHCR_TOKEN in env.
# Usage: ./scripts/build-and-push.sh [tag]           (default: latest)

source "$(dirname "$0")/_common.sh"

TAG="${1:-latest}"
IMAGE="ghcr.io/togodynamicslab/tdvx:$TAG"

command -v docker >/dev/null || die "docker required."

if [[ -n "${GHCR_USER:-}" && -n "${GHCR_TOKEN:-}" ]]; then
    info "Logging into ghcr.io as $GHCR_USER..."
    echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null \
        || die "docker login failed."
fi

info "Building $IMAGE (context: $REPO_ROOT)..."
docker build -t "$IMAGE" "$REPO_ROOT"

info "Pushing $IMAGE..."
docker push "$IMAGE"

info "Done."
info "  Image: $IMAGE"
info "  Use:   TDVX_IMAGE=$IMAGE ./scripts/rent.sh"
