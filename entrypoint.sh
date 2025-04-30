#!/usr/bin/env bash
set -e
source activate universe

# one-off tasks
if [ ! -d data/voicebank_demand/48k ]; then
  echo "==> first-run setup (this is cached on persistent volume)"
  bash models/universe/data/download.sh
  bash models/universe/data/prepare.sh
fi

exec "$@"
