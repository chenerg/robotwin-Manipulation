"""Geometry/protocol/control tests; these do not stand in for SAPIEN validation."""

import asyncio
import copy
import json
import queue
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from teleop_web.controller import Controller
from teleop_web.geometry import (
    camera_axes,
    ee_to_tcp,
    local_target,
    pose_error,
    quat_matrix,
    tcp_to_ee,
)
from teleop_web.protocol import pack_frame, validate_move
from teleop_web.server import TeleopServer
from teleop_web.session import (
    CONFIG_ROOT,
    RobotwinSession,
    catalog,
    load_task_args,
    normalize_embodiment,
    validate_scene,
)
from teleop_web.worker import capture_frame


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
        "frame": "tcp_local",
        "left": copy.deepcopy(data) if left else None,
        "right": copy.deepcopy(data) if right else None,
    }


class GeometryTests(unittest.TestCase):
    def test_translation_uses_current_local_axes(self):
        # Local +X of a TCP rotated 90 degrees around world Z is world +Y.
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

    def test_rotation_holds_tcp_and_moves_original_ee(self):
        ee_initial = [0.4, 0.2, 0.8, 1, 0, 0, 0]
        tcp_initial = ee_to_tcp(ee_initial)
        target = local_target(tcp_initial, [0, 0, 0], [0, 0, 90])
        np.testing.assert_equal(target[:3], tcp_initial[:3])
        ee_target = tcp_to_ee(target)
        np.testing.assert_allclose(ee_target[:3], [0.52, 0.08, 0.8], atol=1e-12)
        np.testing.assert_allclose(ee_to_tcp(ee_target), target, atol=1e-12)

    def test_tcp_offset_uses_pose_orientation_and_roundtrips(self):
        ee = [0.4, 0.2, 0.8, 2**-0.5, 0, 0, 2**-0.5]
        tcp = ee_to_tcp(ee)
        np.testing.assert_allclose(tcp[:3], [0.4, 0.32, 0.8], atol=1e-12)
        np.testing.assert_allclose(tcp_to_ee(tcp), ee, atol=1e-12)
        # Position error must be measured at TCP, including orientation's offset.
        rotated_ee = local_target(ee, [0, 0, 0], [0, 0, 90])
        self.assertAlmostEqual(
            pose_error(tcp, ee_to_tcp(rotated_ee))["position_m"], 0.12 * 2**0.5
        )

    def test_sign_invariant_rotation_error(self):
        a = [0, 0, 0, 1, 0, 0, 0]
        self.assertEqual(pose_error(a, [0, 0, 0, -1, 0, 0, 0])["rotation_deg"], 0)

    def test_camera_projection_uses_world_to_camera_cv(self):
        extrinsic = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3]])
        axes = camera_axes([1, 0, 1, 1, 0, 0, 0], extrinsic)
        np.testing.assert_allclose(axes["origin"], [1, 3, 4])
        np.testing.assert_allclose(axes["basis"], extrinsic[:3, :3])


class ProtocolTests(unittest.TestCase):
    def test_reject_legacy_ee_frame_instead_of_moving_with_wrong_pivot(self):
        with self.assertRaisesRegex(ValueError, "tcp_local"):
            validate_move({**command(), "frame": "endpose_local"})

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


class SceneConfigTests(unittest.TestCase):
    def test_choices_follow_registries_without_requiring_assets(self):
        import yaml

        choices = catalog()
        for key, filename in (
            ("embodiments", "_embodiment_config.yml"),
            ("camera_types", "_camera_config.yml"),
        ):
            with (CONFIG_ROOT / filename).open() as file:
                self.assertEqual(choices[key], sorted(yaml.safe_load(file)))
        self.assertEqual(
            normalize_embodiment(["piper", "franka-panda", "0.6"]),
            ["piper", "franka-panda", 0.6],
        )
        for embodiment in (
            ["unknown"],
            ["piper", "piper"],
            ["piper", "piper", True],
            ["piper", "piper", "nan"],
            ["piper", "piper", 0],
        ):
            with self.subTest(embodiment=embodiment), self.assertRaises(ValueError):
                normalize_embodiment(embodiment)
        with self.assertRaises(ValueError):
            validate_scene(
                "handover_block", "demo_clean", 0, head_camera_type="unknown"
            )

    def test_overrides_load_selected_assets_and_camera_dimensions(self):
        import yaml

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for robot in ("aloha-agilex", "piper", "franka-panda"):
                asset = root / "assets" / "embodiments" / robot
                asset.mkdir(parents=True)
                (asset / "config.yml").write_text(yaml.safe_dump({"name": robot}))
            with patch("teleop_web.session.ROOT", root):
                default = load_task_args("handover_block", "demo_clean")
                self.assertEqual(default["embodiment"], ["aloha-agilex"])
                self.assertEqual(default["camera"]["head_camera_type"], "D435")
                self.assertEqual(
                    (default["head_camera_w"], default["head_camera_h"]), (320, 240)
                )
                overridden = load_task_args(
                    "handover_block",
                    "demo_clean",
                    ["piper", "franka-panda", 0.6],
                    "Large_L515",
                    "L515",
                )
                self.assertEqual(overridden["left_embodiment_config"]["name"], "piper")
                self.assertEqual(
                    overridden["right_embodiment_config"]["name"], "franka-panda"
                )
                self.assertFalse(overridden["dual_arm_embodied"])
                self.assertEqual(overridden["embodiment_dis"], 0.6)
                self.assertEqual(overridden["camera"]["wrist_camera_type"], "L515")
                self.assertEqual(
                    (overridden["head_camera_w"], overridden["head_camera_h"]),
                    (640, 360),
                )
                integrated = load_task_args("handover_block", "demo_clean", ["piper"])
                self.assertTrue(integrated["dual_arm_embodied"])
                self.assertEqual(
                    integrated["left_robot_file"], integrated["right_robot_file"]
                )


class TcpSessionTests(unittest.TestCase):
    def test_control_reads_tcp_and_converts_rotated_target_before_planning(self):
        session = RobotwinSession.__new__(RobotwinSession)
        session.robot = Mock()
        session.robot.get_left_tcp_pose.return_value = [0.52, 0.2, 0.8, 1, 0, 0, 0]
        session.robot.get_right_tcp_pose.return_value = [0, 0, 0.8, 1, 0, 0, 0]
        actual = session.poses()
        target = local_target(actual["left"], [0, 0, 0], [0, 0, 90])
        session.robot.left_entity.get_qpos.return_value = np.zeros(8)
        requests = session.plan_requests({"left": target})
        ee_target = session.robot._trans_from_gripper_to_endlink.call_args.args[0]
        np.testing.assert_allclose(ee_target[:3], [0.52, 0.08, 0.8], atol=1e-12)
        np.testing.assert_allclose(ee_to_tcp(ee_target), target, atol=1e-12)
        self.assertEqual(
            session.robot._trans_from_gripper_to_endlink.call_args.kwargs,
            {"arm_tag": "left"},
        )
        np.testing.assert_equal(requests["left"]["qpos"], np.zeros(8))
        session.robot.right_entity.get_qpos.assert_not_called()

    def test_frame_pose_and_axes_use_tcp_from_same_observation(self):
        ee = [0.4, 0.2, 0.8, 2**-0.5, 0, 0, 2**-0.5]
        extrinsic = np.eye(4)
        camera = {
            "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            "intrinsic_cv": np.eye(3),
            "extrinsic_cv": extrinsic,
        }
        session = Mock(sim_time=1.0)
        session.observe.return_value = {
            "endpose": {"left_endpose": ee, "right_endpose": ee},
            "observation": {
                name: camera for name in ("head_camera", "left_camera", "right_camera")
            },
        }
        session.gripper.return_value = session.measured_gripper.return_value = 0.4
        session.success.return_value = False
        encoder = SimpleNamespace(
            COLOR_RGB2BGR=1,
            IMWRITE_JPEG_QUALITY=2,
            cvtColor=lambda image, _: image,
            imencode=lambda *args: (True, np.array([1, 2, 3], dtype=np.uint8)),
        )
        with patch.dict("sys.modules", {"cv2": encoder}):
            packet = capture_frame(session, "scene", 1, "ready", 80)
        size = struct.unpack("!I", packet[:4])[0]
        metadata = json.loads(packet[4 : 4 + size])
        self.assertEqual(metadata["pose_frame"], "tcp")
        np.testing.assert_allclose(metadata["poses"]["left"][:3], [0.4, 0.32, 0.8])
        axes = metadata["cameras"][0]["axes"]["left"]
        np.testing.assert_allclose(axes["origin"], [0.4, 0.32, 0.8])
        np.testing.assert_allclose(axes["basis"], quat_matrix(ee[3:]), atol=1e-12)
        session.poses.assert_not_called()


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

    async def test_history_retains_command_and_outcome_on_reconnect(self):
        self.app.owner = "controller"
        original = command(right=True, rotation=(1, 2, 3), gripper=0.4)
        self.app.dispatch("controller", original)
        original["left"]["translation_m"][0] = 999
        self.app.record_event(
            {
                "type": "state",
                "command_id": "test-command",
                "phase": "accepted",
                "base_poses": {"left": [0, 0, 0, 1, 0, 0, 0]},
                "targets": {"left": [0.01, 0, 0, 1, 0, 0, 0]},
            }
        )
        event = {
            "type": "state",
            "command_id": "test-command",
            "phase": "failed",
            "actual_poses": {"left": [0.005, 0, 0, 1, 0, 0, 0]},
            "errors": {"left": {"position_m": 0.005, "rotation_deg": 0}},
        }
        self.app.record_event(event)
        self.app.record_event(event)  # Repeated outcomes must not duplicate a row.
        async with connect(f"ws://127.0.0.1:{self.port}/ws/control") as connection:
            state = json.loads(await connection.recv())["state"]
        self.assertEqual(len(state["history"]), 1)
        row = state["history"][0]
        self.assertEqual(row["left"]["translation_m"], [0.01, 0, 0])
        self.assertEqual(row["left"]["rotation_deg"], [1, 2, 3])
        self.assertEqual(row["right"]["gripper"], 0.4)
        self.assertEqual(row["phase"], "failed")
        self.assertIn("base_poses", row)
        self.assertIn("actual_poses", row)
        self.assertNotIn("history", self.app.snapshot())

    async def test_error_terminates_active_history_but_busy_rejection_adds_no_row(self):
        self.app.owner = "controller"
        self.app.dispatch("controller", command())
        with self.assertRaises(ValueError):
            self.app.dispatch("controller", {**command(), "command_id": "busy-command"})
        self.assertEqual(len(self.app.history), 1)
        self.app.record_event(
            {"type": "state", "phase": "error", "error": "worker exited"}
        )
        self.assertEqual(self.app.history_entry["phase"], "failed")
        self.assertEqual(self.app.history_entry["error"], "worker exited")

    async def test_reset_validates_and_passes_registry_overrides(self):
        self.app.owner = "controller"
        message = {
            "type": "reset",
            "scene_id": "test-scene",
            "embodiment": ["piper", "franka-panda", 0.6],
            "head_camera_type": "L515",
            "wrist_camera_type": "Large_L515",
        }
        with patch.object(self.app, "reset", new_callable=AsyncMock) as reset:
            self.app.dispatch("controller", message)
            await self.app.reset_task
            settings = reset.call_args.args[0]
            self.assertEqual(settings["embodiment"], message["embodiment"])
            self.assertEqual(settings["head_camera_type"], "L515")
            self.assertEqual(settings["wrist_camera_type"], "Large_L515")
        self.app.resetting = False
        with self.assertRaises(ValueError):
            self.app.dispatch(
                "controller", {**message, "embodiment": ["not-registered"]}
            )
        with self.assertRaises(ValueError):
            self.app.dispatch(
                "controller", {**message, "head_camera_type": "not-registered"}
            )

    async def test_new_scene_clears_history(self):
        self.app.owner = "controller"
        self.app.dispatch("controller", command())
        old_scene = self.app.scene_id
        with patch.object(self.app, "context"):
            self.app.start_worker()
        self.assertNotEqual(self.app.scene_id, old_scene)
        self.assertEqual(self.app.history, {})
        self.assertIsNone(self.app.history_entry)
        self.app.process = None


if __name__ == "__main__":
    unittest.main()
