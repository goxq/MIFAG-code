#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/eval_seen.sh CKPT_PATH [extra eval_mifag.py args]"
  exit 1
fi

CKPT_PATH="$1"
shift

python eval_mifag.py \
  --config configs/mifag_seen.yaml \
  --ckpt_path "$CKPT_PATH" \
  "$@"
