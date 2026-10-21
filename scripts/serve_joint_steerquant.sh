#!/usr/bin/env bash
# Start only the model server. No ROS, camera or robot processes are started.
set -euo pipefail
STEERQUANT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STEERQUANT_TASK="${1:-pack3}"
if (($#)); then shift; fi
case "$STEERQUANT_TASK" in
  pack3|pack) STEERQUANT_TASK=pack ;;
  stack3|stack) STEERQUANT_TASK=stack ;;
  *) echo 'Usage: bash scripts/serve_joint_steerquant.sh {pack3|stack3} [w4a8|w4a4|bf16] [server options]' >&2; exit 2 ;;
esac
STEERQUANT_MODE=w4a8
if [[ "${1:-}" == w4a8 || "${1:-}" == bf16 || "${1:-}" == w4a4 ]]; then
  STEERQUANT_MODE="$1"
  shift
fi
exec bash "$STEERQUANT_ROOT/bash/run_robot_profile.sh" server "$STEERQUANT_TASK" "$STEERQUANT_MODE" "$@"
