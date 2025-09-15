#!/bin/zsh

# Usage: ./download_youtube.sh <youtube_url>
# Downloads a YouTube video to temp/ in the MAXIMUM available quality
# using yt-dlp (bestvideo+bestaudio) with ffmpeg merge.

set -euo pipefail

PROJECT_DIR="/Users/edcher/Documents/GitHub/YouTube"
VENV_BIN="$PROJECT_DIR/venv/bin"
OUT_DIR="$PROJECT_DIR/temp"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <youtube_url>"
  exit 1
fi

URL="$1"

# Prefer yt-dlp from the project venv, fallback to system yt-dlp, then python -m yt_dlp
if [ -x "$VENV_BIN/yt-dlp" ]; then
  YT_DLP_BIN="$VENV_BIN/yt-dlp"
  USE_PY_MODULE=0
elif command -v yt-dlp >/dev/null 2>&1; then
  YT_DLP_BIN="$(command -v yt-dlp)"
  USE_PY_MODULE=0
elif [ -x "$VENV_BIN/python" ]; then
  PY_BIN="$VENV_BIN/python"
  USE_PY_MODULE=1
else
  echo "yt-dlp not found. Install into venv: $VENV_BIN/pip install -U yt-dlp"
  exit 1
fi

# ffmpeg is required to merge bestvideo+bestaudio without re-encoding
FFMPEG_OPTS=()
if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_LOC="$(command -v ffmpeg)"
  FFMPEG_OPTS=(--ffmpeg-location "$FFMPEG_LOC")
else
  echo "Warning: ffmpeg not found. Install (e.g. brew install ffmpeg) for proper merging."
fi

mkdir -p "$OUT_DIR"

# Highest available quality: prefer separate bestvideo+bestaudio, fallback to best
FORMAT='bv*+ba/b'
# Use MKV container to support modern codecs (AV1/VP9 + Opus) without re-encoding
MERGE_CONTAINER='mkv'

# Prefer authenticated cookies to bypass YouTube bot checks and age/region gates
COOKIES_ARGS=()
# Allow override via env var YT_COOKIES_FROM_BROWSER (e.g., chrome, chromium)
if [ -n "${YT_COOKIES_FROM_BROWSER:-}" ]; then
  COOKIES_ARGS=(--cookies-from-browser "$YT_COOKIES_FROM_BROWSER")
else
  # Default to Chrome if its profile directory exists
  if [ -d "$HOME/Library/Application Support/Google/Chrome" ]; then
    COOKIES_ARGS=(--cookies-from-browser chrome)
  fi
fi

echo "Downloading in maximum quality to: $OUT_DIR"
if [ "${USE_PY_MODULE:-0}" -eq 1 ]; then
  "$PY_BIN" -m yt_dlp \
    -f "$FORMAT" \
    --merge-output-format "$MERGE_CONTAINER" \
    --geo-bypass \
    --force-overwrites \
    -N 8 \
    -o "$OUT_DIR/%(title)s.%(ext)s" \
    "${FFMPEG_OPTS[@]}" \
    "${COOKIES_ARGS[@]}" \
    "$URL"
else
  "$YT_DLP_BIN" \
    -f "$FORMAT" \
    --merge-output-format "$MERGE_CONTAINER" \
    --geo-bypass \
    --force-overwrites \
    -N 8 \
    -o "$OUT_DIR/%(title)s.%(ext)s" \
    "${FFMPEG_OPTS[@]}" \
    "${COOKIES_ARGS[@]}" \
    "$URL"
fi

echo "Done."
