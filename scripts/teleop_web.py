#!/usr/bin/env python3
"""Start browser-based human control without a policy server or expert rollout."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--embodiment",
        nargs="+",
        metavar="VALUE",
        help="Registered NAME, or LEFT RIGHT SPACING_M; defaults to task config",
    )
    parser.add_argument("--head-camera-type", help="Camera registry preset, e.g. L515")
    parser.add_argument("--wrist-camera-type", help="Camera registry preset, e.g. L515")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.005,
        help="Arrival tolerance in meters",
    )
    parser.add_argument(
        "--rotation-tolerance",
        type=float,
        default=3.0,
        help="Arrival tolerance in degrees",
    )
    args = parser.parse_args()
    from teleop_web.session import normalize_embodiment, validate_scene

    try:
        args.embodiment = normalize_embodiment(args.embodiment)
        validate_scene(
            args.task_name,
            args.task_config,
            args.seed,
            args.embodiment,
            args.head_camera_type,
            args.wrist_camera_type,
        )
        import math

        if not math.isfinite(args.fps) or not 0 < args.fps <= 60:
            raise ValueError("--fps must be in (0, 60]")
        if not 1 <= args.jpeg_quality <= 100 or not 1 <= args.port <= 65535:
            raise ValueError("Invalid JPEG quality or port")
        if not all(
            math.isfinite(x) and x > 0
            for x in (args.position_tolerance, args.rotation_tolerance)
        ):
            raise ValueError("Arrival tolerances must be positive and finite")
    except ValueError as exc:
        parser.error(str(exc))
    from teleop_web.server import run_server

    settings = {
        key: value for key, value in vars(args).items() if key not in {"host", "port"}
    }
    asyncio.run(run_server(settings, args.host, args.port))


if __name__ == "__main__":
    main()
