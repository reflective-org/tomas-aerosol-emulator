#!/usr/bin/env bash
# Fetch the axial_aug5 model weights (~113 MB) from the GitHub release.
# The file exceeds GitHub's 100 MB in-repo limit, so it ships as a release
# asset instead of being tracked in git.
set -euo pipefail

REPO="MShreyas1/tomas-aerosol-emulator"
TAG="${TAG:-axial-aug5-weights}"
DEST="checkpoints/axial_aug5/best_model.pt"

if [ -f "$DEST" ]; then
    echo "$DEST already exists; delete it first to re-download."
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
URL="https://github.com/$REPO/releases/download/$TAG/best_model.pt"
echo "Downloading $URL"
curl -fL --retry 3 -o "$DEST" "$URL"
echo "Saved to $DEST"
