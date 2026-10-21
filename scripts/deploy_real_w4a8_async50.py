#!/usr/bin/env python3
"""Compatibility entry for the portable W4A8 live client (shadow by default).

Use configs/robot_{pack,stack}_w4a8_live.example.yaml. Historical BF16,
recorded-episode and W4A4 paths are outside this public entry.
"""
from fastwam_steerquant.real_robot.run import main


if __name__ == "__main__":
    main()
