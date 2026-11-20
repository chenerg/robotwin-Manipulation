"""SAPIEN owner process. Rendering and physics always run on this main thread."""

import multiprocessing as mp
import os
import queue
import signal
import time
import traceback

import numpy as np

from .controller import Controller
from .geometry import camera_axes, ee_to_tcp
from .protocol import CAMERAS, pack_frame


def latest_put(channel, item):
    try:
        channel.put_nowait(item)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(item)
        except queue.Full:
            pass


def capture_frame(session, scene_id, frame_id, phase, quality):
    import cv2

    observation = session.observe()
    # Convert the observation's EE poses so overlays stay aligned to this frame.
    poses = {
        arm: ee_to_tcp(observation["endpose"][f"{arm}_endpose"])
        for arm in ("left", "right")
    }
    metadata = {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "sim_time": session.sim_time,
        "captured_at": time.time(),
        "phase": phase,
        "poses": poses,
        "pose_frame": "tcp",
        "grippers": {arm: session.gripper(arm) for arm in poses},
        "gripper_positions": {arm: session.measured_gripper(arm) for arm in poses},
        "task_success": session.success(),
        "cameras": [],
    }
    jpegs = []
    for name in CAMERAS:
        camera = observation["observation"][name]
        image = camera["rgb"]
        # Camera.get_rgb returns RGB; OpenCV's JPEG encoder expects BGR.
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            raise RuntimeError(f"Could not encode {name}")
        selected = ("left", "right") if name == "head_camera" else (name.split("_")[0],)
        metadata["cameras"].append(
            {
                "name": name,
                "width": image.shape[1],
                "height": image.shape[0],
                "jpeg_bytes": encoded.nbytes,
                "near": 0.1,
                "far": 100.0,
                "intrinsic": np.asarray(camera["intrinsic_cv"]).tolist(),
                "axes": {
                    arm: camera_axes(poses[arm], camera["extrinsic_cv"])
                    for arm in selected
                },
            }
        )
        jpegs.append(encoded.tobytes())
    return pack_frame(metadata, jpegs)


def run_worker(settings, scene_id, commands, events, frames):
    mp.set_start_method("spawn", force=True)
    if os.name == "posix":
        os.setsid()  # The server can clean up this scene and its planner children.

    def terminate(_signum, _frame):
        raise SystemExit

    signal.signal(signal.SIGTERM, terminate)
    session = controller = None

    def emit(event):
        events.put({"scene_id": scene_id, **event})

    try:
        from .session import RobotwinSession

        emit({"type": "state", "phase": "starting"})
        session = RobotwinSession(
            settings["task_name"],
            settings["task_config"],
            settings["seed"],
            settings.get("embodiment"),
            settings.get("head_camera_type"),
            settings.get("wrist_camera_type"),
        )
        controller = Controller(
            session,
            emit,
            settings["position_tolerance"],
            settings["rotation_tolerance"],
        )
        emit({"type": "state", "phase": "ready", "detail": "Scene ready"})
        next_tick = next_frame = time.monotonic()
        frame_id = 0
        running = True
        while running:
            for _ in range(16):
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                if command["type"] == "shutdown":
                    running = False
                    break
                if command.get("scene_id") != scene_id:
                    continue
                if command["type"] == "stop":
                    controller.stop()
                elif command["type"] == "move":
                    try:
                        controller.begin(command)
                    except ValueError as exc:
                        emit(
                            {
                                "type": "rejected",
                                "phase": controller.phase,
                                "command_id": command["command_id"],
                                "error": str(exc),
                            }
                        )
            if not running:
                break
            now = time.monotonic()
            if now >= next_tick:
                controller.update()
                if not controller.freeze_physics:
                    session.step()
                # Drop wall-clock backlog rather than advancing a burst of stale steps.
                next_tick = max(next_tick + session.dt, time.monotonic())
            if now >= next_frame:
                latest_put(
                    frames,
                    capture_frame(
                        session,
                        scene_id,
                        frame_id,
                        controller.phase,
                        settings["jpeg_quality"],
                    ),
                )
                frame_id += 1
                next_frame = time.monotonic() + 1 / settings["fps"]
            time.sleep(max(0, min(next_tick, next_frame) - time.monotonic()))
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 -- report simulator/import failures across the process boundary
        traceback.print_exc()
        emit(
            {"type": "state", "phase": "error", "error": f"{type(exc).__name__}: {exc}"}
        )
    finally:
        if controller is not None:
            controller.close()
        for process in mp.active_children():
            process.terminate()
            process.join(timeout=1)
        if session is not None:
            session.close()
