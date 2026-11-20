"""Task configuration and simulator access, with lazy simulator imports."""

import copy
import importlib
import math
import os
import random
from pathlib import Path

import numpy as np
import yaml

from .geometry import tcp_to_ee

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "env_cfg" / "task_config"


def registry(name):
    with (CONFIG_ROOT / name).open() as file:
        return yaml.safe_load(file)


def catalog():
    return {
        "tasks": sorted(
            p.stem
            for p in (ROOT / "envs").glob("*.py")
            if not p.stem.startswith("_") and p.stem != "model_utils"
        ),
        "task_configs": sorted(
            p.stem for p in CONFIG_ROOT.glob("*.yml") if not p.stem.startswith("_")
        ),
        "embodiments": sorted(registry("_embodiment_config.yml")),
        "camera_types": sorted(registry("_camera_config.yml")),
    }


def normalize_embodiment(value):
    """Use RoboTwin's [dual robot] or [left robot, right robot, spacing] form."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) not in (1, 3):
        raise ValueError("embodiment must contain NAME or LEFT RIGHT SPACING_M")
    names = value[:1] if len(value) == 1 else value[:2]
    choices = registry("_embodiment_config.yml")
    if any(not isinstance(name, str) or name not in choices for name in names):
        raise ValueError("Unknown embodiment; choose a name from the registry")
    result = list(names)
    if len(value) == 3:
        try:
            spacing = float(value[2])
        except (TypeError, ValueError):
            raise ValueError("Embodiment spacing must be positive and finite") from None
        if isinstance(value[2], bool) or not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("Embodiment spacing must be positive and finite")
        result.append(spacing)
    return result


def validate_scene(
    task_name,
    task_config,
    seed,
    embodiment=None,
    head_camera_type=None,
    wrist_camera_type=None,
):
    choices = catalog()
    if task_name not in choices["tasks"]:
        raise ValueError(f"Unknown task_name: {task_name}")
    if task_config not in choices["task_configs"]:
        raise ValueError(f"Unknown task_config: {task_config}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    normalize_embodiment(embodiment)
    for camera in (head_camera_type, wrist_camera_type):
        if camera is not None and (
            not isinstance(camera, str) or camera not in choices["camera_types"]
        ):
            raise ValueError(f"Unknown camera type: {camera}")


def load_task_args(
    task_name,
    task_config,
    embodiment=None,
    head_camera_type=None,
    wrist_camera_type=None,
):
    with (CONFIG_ROOT / f"{task_config}.yml").open() as file:
        args = copy.deepcopy(yaml.safe_load(file))
    with (CONFIG_ROOT / "_embodiment_config.yml").open() as file:
        embodiments = yaml.safe_load(file)
    types = normalize_embodiment(
        embodiment if embodiment is not None else args["embodiment"]
    )
    args["embodiment"] = types
    for arm, robot in (
        ("left", types[0]),
        ("right", types[0] if len(types) == 1 else types[1]),
    ):
        robot_path = Path(embodiments[robot]["file_path"])
        if not robot_path.is_absolute():
            robot_path = ROOT / robot_path
        config = robot_path / "config.yml"
        if not config.is_file():
            raise FileNotFoundError(
                f"Missing robot asset: {config}. Download RoboTwin embodiment assets first."
            )
        args[f"{arm}_robot_file"] = str(robot_path)
        with config.open() as file:
            args[f"{arm}_embodiment_config"] = yaml.safe_load(file)
    args["dual_arm_embodied"] = len(types) == 1
    if len(types) == 3:
        args["embodiment_dis"] = types[2]
    args.update(
        task_name=task_name,
        task_config=task_config,
        render_freq=0,
        save_data=False,
        collect_data=False,
        eval_mode=False,
        eval_video_save_dir=None,
        need_plan=True,
    )
    args["camera"].update(collect_head_camera=True, collect_wrist_camera=True)
    for key, value in (
        ("head_camera_type", head_camera_type),
        ("wrist_camera_type", wrist_camera_type),
    ):
        if value is not None:
            if value not in registry("_camera_config.yml"):
                raise ValueError(f"Unknown camera type: {value}")
            args["camera"][key] = value
    args["data_type"] = {"rgb": True, "endpose": True, "qpos": True}
    with (CONFIG_ROOT / "_camera_config.yml").open() as file:
        camera = yaml.safe_load(file)[args["camera"]["head_camera_type"]]
    args.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
    return args


class RobotwinSession:
    def __init__(
        self,
        task_name,
        task_config,
        seed,
        embodiment=None,
        head_camera_type=None,
        wrist_camera_type=None,
    ):
        self.env = None
        os.chdir(ROOT)  # Existing task/asset code uses repository-relative paths.
        args = load_task_args(
            task_name, task_config, embodiment, head_camera_type, wrist_camera_type
        )
        random.seed(seed)
        module = importlib.import_module(f"envs.{task_name}")
        self.env = getattr(module, task_name)()
        self.env.setup_demo(now_ep_num=0, seed=seed, **args)
        self.robot = self.env.robot
        self.dt = float(self.env.scene.get_timestep())
        self.sim_time = 0.0

    def poses(self):
        return {
            arm: getattr(self.robot, f"get_{arm}_tcp_pose")()
            for arm in ("left", "right")
        }

    def measured_joints(self, arm):
        return np.asarray(
            getattr(self.robot, f"get_{arm}_arm_real_jointState")()[:-1], dtype=float
        )

    def gripper(self, arm):
        return float(getattr(self.robot, f"get_{arm}_gripper_val")())

    def measured_gripper(self, arm):
        entity = getattr(self.robot, f"{arm}_entity")
        joint = getattr(self.robot, f"{arm}_gripper")[0][0]
        index = entity.get_active_joints().index(joint)
        low, high = getattr(self.robot, f"{arm}_gripper_scale")
        return float(np.clip((entity.get_qpos()[index] - low) / (high - low), 0, 1))

    def plan_requests(self, targets):
        """Snapshot every scene read on the simulation thread before planning."""
        requests = {}
        for arm, target in targets.items():
            requests[arm] = {
                "qpos": getattr(self.robot, f"{arm}_entity").get_qpos().copy(),
                "target_pose": self.robot._trans_from_gripper_to_endlink(
                    tcp_to_ee(target), arm_tag=arm
                ),
                "arms_tag": arm,
                "constraint_pose": None,
            }
        return requests

    def plan(self, requests):
        """Planning thread: no scene/entity/render access, only immutable snapshots."""
        plans = {}
        for arm, request in requests.items():
            if self.robot.communication_flag:
                connection = getattr(self.robot, f"{arm}_conn")
                connection.send({"cmd": "plan_path", **request})
                plans[arm] = connection.recv()
            else:
                planner = getattr(self.robot, f"{arm}_planner")
                plans[arm] = planner.plan_path(
                    request["qpos"],
                    request["target_pose"],
                    constraint_pose=None,
                    arms_tag=arm,
                )
        return plans

    def drive(self, arm, position, velocity):
        self.robot.set_arm_joints(position, velocity, arm)

    def set_gripper(self, arm, value):
        self.robot.set_gripper(value, arm)

    def step(self):
        entities = {
            id(entity): entity
            for entity in (self.robot.left_entity, self.robot.right_entity)
        }
        for entity in entities.values():
            self.robot._entity_qf(entity)
        self.env.scene.step()
        self.sim_time += self.dt

    def observe(self):
        return self.env.get_obs()

    def success(self):
        return bool(self.env.check_success())

    def close(self):
        if self.env is not None:
            self.env.close_env()
