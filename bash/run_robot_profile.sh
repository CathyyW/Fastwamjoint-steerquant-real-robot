#!/usr/bin/env bash
# Default client uses recorded observations; live-client is shadow-only as well.
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo 'Usage: bash bash/run_robot_profile.sh {server|client|live-client} {pack|stack} {bf16|w4a8|w4a4} [options]' >&2
  exit 2
fi
role="$1"; task="$2"; mode="$3"; shift 3
case "$task" in pack|stack) ;; *) echo 'Expected pack or stack' >&2; exit 2;; esac
case "$mode" in bf16) hz=25;; w4a8|w4a4) hz=40;; *) echo 'Expected bf16, w4a8 or w4a4' >&2; exit 2;; esac
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
: "${FASTWAM_AUTHKEY:?Set the same FASTWAM_AUTHKEY in both terminals}"
config="configs/robot_${task}_${mode}_async${hz}.yaml"
output="outputs/profile_${task}_${mode}_$(date +%Y%m%d_%H%M%S_%N)"
case "$role" in
  server)
    if [ "$mode" = w4a8 ]; then
      exec bash bash/with_real_w4a8_runtime.sh python -u scripts/serve_robot_profile.py \
        --config "$config" --report "${output}_startup.json" "$@"
    fi
    unset FASTWAM_RUNTIME_PROFILE
    exec python -u scripts/serve_robot_profile.py --config "$config" --report "${output}_startup.json" "$@"
    ;;
  client)
    exec python -u scripts/mock_w4a8_async50.py --config "$config" --max-steps 128 --output-dir "$output" "$@"
    ;;
  live-client)
    # Imports existing controller read-only; starts no ROS launch or camera driver.
    export PYTHONPATH="$root/src:/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/.worktrees/joint-asi-async50/src${PYTHONPATH:+:$PYTHONPATH}"
    exec python -u scripts/deploy_real_w4a8_async50.py --config "$config" --max-steps 128 \
      --output-dir "$output" "$@" --no-execute-actions --no-move-to-initial
    ;;
  *) echo 'Expected server, client or live-client' >&2; exit 2;;
esac
