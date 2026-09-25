#!/usr/bin/env bash
# Pull the manuscript folder from Google Drive into data/raw/.
#
# Run this on a machine with normal internet access (your laptop or training
# box). Sandboxed/CI environments often block drive.google.com at the TLS
# layer, in which case use the --ids mode below and fetch file-by-file, or
# simply download the ZIP from the Drive UI and unzip it into data/raw/.
#
# Usage:
#   ./scripts/download_from_drive.sh                      # whole folder
#   ./scripts/download_from_drive.sh --ids id1 id2        # specific files
#   ./scripts/download_from_drive.sh --out data/raw/books
set -euo pipefail

FOLDER_URL="https://drive.google.com/drive/folders/1gKv076l13RyRWt-8InkgJm9H8z4qtAru"
OUT="data/raw"
MODE="folder"
IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --ids) MODE="ids"; shift; while [[ $# -gt 0 && "$1" != --* ]]; do IDS+=("$1"); shift; done ;;
    --url) FOLDER_URL="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! python3 -c "import gdown" >/dev/null 2>&1; then
  echo "installing gdown..."
  pip install --quiet gdown || pip install --quiet --break-system-packages gdown
fi

mkdir -p "$OUT"

if [[ "$MODE" == "folder" ]]; then
  echo "Downloading folder -> $OUT"
  # --remaining-ok is not available on every gdown version; fall back if needed.
  python3 -m gdown --folder --no-cookies "$FOLDER_URL" -O "$OUT" \
    || python3 -m gdown --folder --no-cookies --remaining-ok "$FOLDER_URL" -O "$OUT"
else
  for id in "${IDS[@]}"; do
    echo "Downloading file $id"
    python3 -m gdown --no-cookies --id "$id" -O "$OUT/"
  done
fi

echo
echo "Contents of $OUT:"
find "$OUT" -maxdepth 2 -type f | sed "s|^|  |"
echo
echo "Note: Google Drive zips folders without extensions. If you downloaded the"
echo "folder as a ZIP, unzip it into $OUT and confirm the file extensions are"
echo "preserved (.epub/.docx/.txt)."
