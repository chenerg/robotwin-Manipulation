"""Geometry/protocol/control tests; these do not stand in for SAPIEN validation."""

import asyncio
import copy
import json
import queue
import struct
import threading
import unittest

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from teleop_web.controller import Controller
from teleop_web.geometry import camera_axes, local_target, pose_error, quat_matrix
from teleop_web.protocol import pack_frame, validate_move
from teleop_web.server import TeleopServer
from teleop_web.session import validate_scene


def command(
    left=True, right=False, translation=(0.01, 0, 0), rotation=(0, 0, 0), gripper=None
):
    data = {
        "translation_m": list(translation),
        "rotation_deg": dict(zip(("roll", "pitch", "yaw"), rotation)),
        "gripper": gripper,
    }
    return {
        "type": "move",
        "command_id": "test-command",
        "scene_id": "test-scene",
        "frame": "endpose_local",
        "left": copy.deepcopy(data) if left else None,
        "right": copy.deepcopy(data) if right else None,
    }


class GeometryTests(unittest.TestCase):
    def test_translation_uses_current_local_axes(self):
        # Local +X of an EE rotated 90 degrees around world Z is world +Y.
        s = 2**-0.5
        actual = [1, 2, 3, s, 0, 0, s]
        target = local_target(actual, [0.1, 0, 0], [0, 0, 0])
        np.testing.assert_allclose(target[:3], [1, 2.1, 3], atol=1e-12)

    def test_rotation_is_right_multiplied_and_ordered(self):
        actual = [0, 0, 0, 2**-0.5, 0, 0, 2**-0.5]
        rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        target = local_target(actual, [0, 0, 0], [90, 90, 90])
        np.testing.assert_allclose(
            quat_matrix(target[3:]), rz @ rz @ ry @ rx, atol=1e-12
        )

    def test_rotation_pivots_about_ee_not_tcp(self):
        initial = [0.4, 0.2, 0.8, 1, 0, 0, 0]
        target = local_target(initial, [0, 0, 0], [0, 0, 90])
        np.testing.assert_equal(target[:3], initial[:3])
        tcp = np.asarray(target[:3]) + quat_matrix(target[3:]) @ [0.12, 0, 0]
        np.testing.assert_allclose(tcp, [0.4, 0.32, 0.8], atol=1e-12)

    def test_sign_invariant_rotation_error(self):
        a = [0, 0, 0, 1, 0, 0, 0]
        self.assertEqual(pose_error(a, [0, 0, 0, -1, 0, 0, 0])["rotation_deg"], 0)

    def test_camera_projection_uses_world_to_camera_cv(self):
        extrinsic = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3]])
        axes = camera_axes([1, 0, 1, 1, 0, 0, 0], extrinsic)
        np.testing.assert_allclose(axes["origin"], [1, 3, 4])
        np.testing.assert_allclose(axes["basis"], extrinsic[:3, :3])


class ProtocolTests(unittest.TestCase):
    def test_named_rpy_fields_do_not_depend_on_json_key_order(self):
        msg = command(rotation=(1, 2, 3))
        msg["left"]["rotation_deg"] = {"yaw": 3, "roll": 1, "pitch": 2}
        self.assertEqual(validate_move(msg)["left"]["rotation_deg"], [1, 2, 3])

    def test_reject_bad_inputs_without_executing(self):
        for value in (
            [float("nan"), 0, 0],
            [float("inf"), 0, 0],
            [True, 0, 0],
            ["0", 0, 0],
            [0, 0],
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_move(command(translation=value))
        for gripper in (-0.1, 1.1, True):
            with self.assertRaises(ValueError):
                validate_move(command(gripper=gripper))
        with self.assertRaises(ValueError):
            validate_move(command(left=False))

    def test_frame_bundle_preserves_camera_boundaries(self):
        header = {"scene_id": "s", "cameras": [{"jpeg_bytes": 3}, {"jpeg_bytes": 2}]}
        packet = pack_frame(header, [b"abc", b"de"])
        size = struct.unpack("!I", packet[:4])[0]
        self.assertEqual(json.loads(packet[4 : 4 + size]), header)
        self.assertEqual(packet[4 + size :], b"abcde")

    def test_scene_validation_prevents_arbitrary_import_or_path(self):
        with self.assertRaises(ValueError):
            validate_scene("../other", "demo_clean", 0)
        with self.assertRaises(ValueError):
            validate_scene("handover_block", "../../other", 0)
        for seed in (True, -1, 0.5, 2**32):
            with self.assertRaises(ValueError):
                validate_scene("handover_block", "demo_clean", seed)


class StubSession:
    """Deliberately no physics: exercise controller admission and ordering only."""

    dt = 1 / 250

    def __init__(self):
        self.actual = {arm: [0, 0, 0, 1, 0, 0, 0] for arm in ("left", "right")}
        self.drives = []
        self.gripper_commands = []
        self.plan_results = None
        self.gate = None

    def poses(self):
        return copy.deepcopy(self.actual)

    def measured_joints(self, arm):
        return np.zeros(6)

    def gripper(self, arm):
        return 1.0

    def measured_gripper(self, arm):
        return 1.0

    def plan_requests(self, targets):
        return targets

    def plan(self, targets):
        if self.gate is not None:
            self.gate.wait(timeout=5)
        return (
            self.plan_results
            if self.plan_results is not None
            else {
                arm: {
                    "status": "Success",
                    "position": np.zeros((10, 6)),
                    "velocity": np.zeros((10, 6)),
                }
                for arm in targets
            }
        )

    def drive(self, arm, position, velocity):
        self.drives.append((arm, np.array(position), np.array(velocity)))

    def set_gripper(self, arm, value):
        self.gripper_commands.append((arm, value))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.session = StubSession()
        self.events = []
        self.controller = Controller(self.session, self.events.append)

    def tearDown(self):
        if self.session.gate is not None:
            self.session.gate.set()
        self.controller.close()

    def start(self, msg):
        self.controller.begin(validate_move(msg))
        self.controller.future.result(timeout=2)

    def test_both_arms_must_plan_before_either_moves_or_grips(self):
        self.session.plan_results = {
            "left": {
                "status": "Success",
                "position": np.zeros((2, 6)),
                "velocity": np.zeros((2, 6)),
            },
            "right": {"status": "Fail"},
        }
        self.start(command(right=True, gripper=0))
        self.controller.update()
        self.assertEqual(self.controller.phase, "failed")
        self.assertEqual(self.session.drives, [])
        self.assertEqual(self.session.gripper_commands, [])

    def test_single_arm_never_replans_or_drives_other_arm(self):
        self.start(command())
        self.controller.update()
        self.assertEqual({entry[0] for entry in self.session.drives}, {"left"})
        with self.assertRaises(ValueError):
            self.controller.begin(validate_move(command()))

    def test_next_command_uses_actual_pose_after_partial_motion(self):
        self.start(command())
        self.controller.update()
        self.controller.stop()
        self.session.actual["left"] = [0.4, 0.2, 0.8, 2**-0.5, 0, 0, 2**-0.5]
        self.start(command())
        np.testing.assert_allclose(
            self.controller.targets["left"][:3], [0.4, 0.21, 0.8], atol=1e-12
        )

    def test_stop_during_planning_discards_result(self):
        self.session.gate = threading.Event()
        self.controller.begin(validate_move(command()))
        self.assertTrue(self.controller.freeze_physics)
        self.controller.stop()
        self.assertEqual(self.controller.phase, "stopping")
        self.session.drives.clear()
        self.session.gate.set()
        self.controller.future.result(timeout=2)
        self.controller.update()
        self.assertEqual(self.controller.phase, "stopped")
        self.assertEqual(self.session.drives, [])

    def test_end_of_trajectory_does_not_falsely_report_arrival(self):
        self.start(command(translation=(0.1, 0, 0)))
        for _ in range(600):
            self.controller.update()
        self.assertEqual(self.controller.phase, "failed")
        self.assertAlmostEqual(self.events[-1]["errors"]["left"]["position_m"], 0.1)

    def test_gripper_only_command_needs_no_ik(self):
        self.start(command(translation=(0, 0, 0), gripper=1))
        for _ in range(150):
            self.controller.update()
        self.assertEqual(self.controller.phase, "completed")
        self.assertEqual(self.session.drives, [])

    def test_contact_limited_closure_is_not_an_ik_failure(self):
        self.session.measured_gripper = lambda arm: 0.4
        self.start(command(translation=(0, 0, 0), gripper=0))
        for _ in range(150):
            self.controller.update()
        self.assertEqual(self.controller.phase, "completed")
        self.assertAlmostEqual(self.events[-1]["errors"]["left"]["gripper"], 0.4)

    def test_shorter_arm_holds_while_longer_arm_finishes(self):
        self.session.plan_results = {
            arm: {
                "status": "Success",
                "position": np.full((length, 6), i),
                "velocity": np.ones((length, 6)),
            }
            for arm, length, i in [("left", 5, 1), ("right", 100, 2)]
        }
        self.start(command(right=True))
        for _ in range(15):
            self.controller.update()
        left, right = self.session.drives[-2:]
        np.testing.assert_equal(left[2], np.zeros(6))
        np.testing.assert_equal(right[2], np.ones(6))


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = TeleopServer(
            {"task_name": "handover_block", "task_config": "demo_clean", "seed": 0}
        )
        self.app.scene_id = "test-scene"
        self.app.phase = "ready"
        self.app.commands = queue.Queue()
        self.server = await serve(
            self.app.handler, "127.0.0.1", 0, process_request=self.app.http
        )
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()

    async def test_control_lease_busy_stale_and_duplicate_commands(self):
        url = f"ws://127.0.0.1:{self.port}/ws/control"
        async with connect(url) as first, connect(url) as second:
            hello1 = json.loads(await first.recv())
            await second.recv()
            await first.send(json.dumps({"type": "claim"}))
            self.assertEqual(
                json.loads(await first.recv())["owner"], hello1["client_id"]
            )
            await second.recv()
            await second.send(json.dumps({"type": "claim"}))
            self.assertEqual(json.loads(await second.recv())["type"], "rejected")
            await first.send(json.dumps(command()))
            self.assertTrue(json.loads(await first.recv())["busy"])
            await first.send(json.dumps({**command(), "command_id": "next"}))
            self.assertIn("not ready", json.loads(await first.recv())["error"])
            self.app.phase = "completed"
            await first.send(json.dumps(command()))
            self.assertIn("Duplicate", json.loads(await first.recv())["error"])
            await first.send(json.dumps({**command(), "scene_id": "old-scene"}))
            self.assertIn("Scene changed", json.loads(await first.recv())["error"])
            self.assertEqual(self.app.commands.qsize(), 1)

    async def test_media_endpoint_delivers_binary_bundle_independently(self):
        async with connect(f"ws://127.0.0.1:{self.port}/ws/frames") as connection:
            packet = pack_frame({"scene_id": "test-scene", "frame_id": 7}, [b"jpeg"])
            async with self.app.frame_changed:
                self.app.packet = packet
                self.app.frame_version += 1
                self.app.frame_changed.notify_all()
            self.assertEqual(
                await asyncio.wait_for(connection.recv(), timeout=2), packet
            )


if __name__ == "__main__":
    unittest.main()
