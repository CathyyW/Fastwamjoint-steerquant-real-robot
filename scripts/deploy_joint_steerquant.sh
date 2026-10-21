#!/usr/bin/env bash
# Hardware-capable client. Motion flags must be supplied explicitly by the user.
# Defaults remain shadow/no-initial-move in the underlying Python entry.
set -euo pipefail
STEERQUANT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STEERQUANT_TASK="${1:-pack3}"
if (($#)); then shift; fi
case "$STEERQUANT_TASK" in
  pack3|pack) STEERQUANT_TASK=pack ;;
  stack3|stack) STEERQUANT_TASK=stack ;;
  *) echo 'Usage: bash scripts/deploy_joint_steerquant.sh {pack3|stack3} [w4a8|w4a4|bf16] [client options]' >&2; exit 2 ;;
esac
STEERQUANT_MODE=w4a8
if [[ "${1:-}" == w4a8 || "${1:-}" == bf16 || "${1:-}" == w4a4 ]]; then
  STEERQUANT_MODE="$1"
  shift
fi
case "$STEERQUANT_MODE" in bf16) STEERQUANT_HZ=25;; w4a8|w4a4) STEERQUANT_HZ=40;; esac
cd "$STEERQUANT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
# Read-only reuse of the audited async runtime, not the joint-async25 model path.
export PYTHONPATH="$STEERQUANT_ROOT/src:/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/.worktrees/joint-asi-async50/src${PYTHONPATH:+:$PYTHONPATH}"
: "${FASTWAM_AUTHKEY:?Set the same FASTWAM_AUTHKEY in both terminals}"
exec python -u scripts/deploy_real_w4a8_async50.py \
  --config "configs/robot_${STEERQUANT_TASK}_${STEERQUANT_MODE}_async${STEERQUANT_HZ}.yaml" "$@"
