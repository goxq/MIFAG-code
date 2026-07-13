#!/usr/bin/env bash
set -euo pipefail

python train_mifag.py \
  --yaml configs/mifag_seen.yaml \
  --save_dir runs/train \
  --name mifag_seen \
  "$@"
