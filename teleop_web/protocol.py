"""Small JSON commands and synchronized, binary three-camera frame bundles."""

import json
import struct

from .geometry import finite_vector

CAMERAS = ("head_camera", "left_camera", "right_camera")
BUSY_PHASES = {
    "starting",
    "accepted",
    "planning",
    "executing",
    "settling",
    "stopping",
    "resetting",
}


def validate_move(message):
    if message.get("frame") != "endpose_local":
        raise ValueError("frame must be endpose_local")
    command_id = message.get("command_id")
    if not isinstance(command_id, str) or not 1 <= len(command_id) <= 128:
        raise ValueError(
            "command_id must be a nonempty string (at most 128 characters)"
        )
    result = {
        "type": "move",
        "scene_id": message.get("scene_id"),
        "command_id": command_id,
        "frame": "endpose_local",
    }
    for arm in ("left", "right"):
        data = message.get(arm)
        if data is None:
            result[arm] = None
            continue
        if not isinstance(data, dict):
            raise TypeError(f"{arm} must be an object or null")
        rotation = data.get("rotation_deg")
        if not isinstance(rotation, dict) or set(rotation) != {"roll", "pitch", "yaw"}:
            raise ValueError("rotation_deg requires roll, pitch, yaw")
        angles = finite_vector(
            [rotation[k] for k in ("roll", "pitch", "yaw")], 3, "rotation_deg"
        )
        translation = finite_vector(data.get("translation_m"), 3, "translation_m")
        gripper = data.get("gripper")
        if gripper is not None:
            gripper = float(finite_vector([gripper], 1, "gripper")[0])
            if not 0 <= gripper <= 1:
                raise ValueError("gripper must be in [0, 1]")
        result[arm] = {
            "translation_m": translation.tolist(),
            "rotation_deg": angles.tolist(),
            "gripper": gripper,
        }
    if result["left"] is None and result["right"] is None:
        raise ValueError("Select at least one arm")
    return result


def pack_frame(metadata, jpegs):
    """4-byte big endian JSON size, UTF-8 metadata, then ordered JPEG images."""
    header = json.dumps(
        metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    return struct.pack("!I", len(header)) + header + b"".join(jpegs)
