#!/usr/bin/env bash
# Fetch axial model weights (~113 MB each) from the GitHub releases.
# They exceed GitHub's 100 MB in-repo limit, so they ship as release assets
# instead of being tracked in git.
#
#   scripts/download_axial_checkpoint.sh              # default: axial_cons2
#   MODEL=axial_aug5 scripts/download_axial_checkpoint.sh
set -euo pipefail

REPO="MShreyas1/tomas-aerosol-emulator"
MODEL="${MODEL:-axial_cons2}"
case "$MODEL" in
    axial_aug5)  TAG="${TAG:-axial-aug5-weights}" ;;
    axial_cons2) TAG="${TAG:-axial-cons2-weights}" ;;
    *) echo "unknown MODEL=$MODEL (expected axial_aug5 or axial_cons2)" >&2; exit 1 ;;
esac
DEST="checkpoints/$MODEL/best_model.pt"

if [ -f "$DEST" ]; then
    echo "$DEST already exists; delete it first to re-download."
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
URL="https://github.com/$REPO/releases/download/$TAG/best_model.pt"
echo "Downloading $URL"
curl -fL --retry 3 -o "$DEST" "$URL"
echo "Saved to $DEST"
