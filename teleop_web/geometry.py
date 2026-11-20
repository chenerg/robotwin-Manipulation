"""TCP-local rigid transforms. All simulator quaternions use wxyz."""

import math

import numpy as np


def finite_vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError(f"{name} must contain numbers")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def quat_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError("Invalid wxyz quaternion")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    result = np.array(
        [
            w * v - x * i - y * j - z * k,
            w * i + x * v + y * k - z * j,
            w * j - x * k + y * v + z * i,
            w * k + x * j - y * i + z * v,
        ]
    )
    return result / np.linalg.norm(result)


def _offset_pose(pose, distance):
    result = np.asarray(pose, dtype=float).copy()
    result[:3] += quat_matrix(result[3:]) @ [distance, 0, 0]
    return result.tolist()


def ee_to_tcp(pose):
    """Match Robot._trans_endpose: TCP is 0.12 m along EE's local +X."""
    return _offset_pose(pose, 0.12)


def tcp_to_ee(pose):
    """Convert a TCP target to the EE target expected by RoboTwin planning."""
    return _offset_pose(pose, -0.12)


def local_target(pose, translation, rotation_deg):
    """T_target = T_actual @ delta_T; delta_R = Rz(yaw) Ry(pitch) Rx(roll).

    The input and target are TCP poses. Rotation holds the TCP origin fixed.
    Translation uses the starting TCP axes, independent of this rotation.
    """
    pose = np.asarray(pose, dtype=float)
    roll, pitch, yaw = np.radians(rotation_deg) / 2
    qx = [math.cos(roll), math.sin(roll), 0, 0]
    qy = [math.cos(pitch), 0, math.sin(pitch), 0]
    qz = [math.cos(yaw), 0, 0, math.sin(yaw)]
    delta = quat_product(quat_product(qz, qy), qx)
    with np.errstate(over="ignore", invalid="ignore"):
        target = np.concatenate(
            (
                pose[:3] + quat_matrix(pose[3:]) @ translation,
                quat_product(pose[3:], delta),
            )
        )
    if not np.isfinite(target).all():
        raise ValueError("Movement produced a non-finite target")
    return target.tolist()


def pose_error(target, actual):
    q1, q2 = np.asarray(target[3:]), np.asarray(actual[3:])
    dot = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return {
        "position_m": float(np.linalg.norm(np.asarray(target[:3]) - actual[:3])),
        "rotation_deg": math.degrees(2 * math.acos(min(1.0, dot))),
    }


def camera_axes(pose, extrinsic_cv):
    """World TCP origin/basis -> OpenCV camera frame (right, down, forward)."""
    extrinsic = np.asarray(extrinsic_cv, dtype=float)
    return {
        "origin": (extrinsic[:3, :3] @ pose[:3] + extrinsic[:3, 3]).tolist(),
        "basis": (extrinsic[:3, :3] @ quat_matrix(pose[3:])).tolist(),
    }
