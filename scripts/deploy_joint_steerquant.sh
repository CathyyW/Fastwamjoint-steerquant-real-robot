#!/usr/bin/env bash
# Portable W4A8 client; motion requires CLI flags and terminal confirmation.
set -euo pipefail
STEERQUANT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STEERQUANT_TASK="${1:-pack3}"
if (($#)); then shift; fi
case "$STEERQUANT_TASK" in
  pack3|pack) STEERQUANT_TASK=pack ;;
  stack3|stack) STEERQUANT_TASK=stack ;;
  *) echo 'Usage: bash scripts/deploy_joint_steerquant.sh {pack3|stack3} [w4a8] [client options]' >&2; exit 2 ;;
esac
case "${1:-}" in
  w4a8) shift ;;
  bf16|w4a4) echo 'This public client supports only W4A8' >&2; exit 2 ;;
esac
cd "$STEERQUANT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$STEERQUANT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${FASTWAM_AUTHKEY:?Set the same FASTWAM_AUTHKEY in server and client terminals}"
exec python -u -m fastwam_steerquant.real_robot.run \
  --config "configs/robot_${STEERQUANT_TASK}_w4a8_live.example.yaml" "$@"
