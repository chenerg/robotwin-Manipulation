"""One discrete EE-local command at a time, with interruptible trajectory stepping."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .geometry import local_target, pose_error


class Controller:
    def __init__(self, session, emit, position_tolerance=0.005, rotation_tolerance=3.0):
        self.session = session
        self.emit = emit
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="teleop-planner"
        )
        self.active = None
        self.phase = "ready"
        self.future = None
        self.stopping = False

    def event(self, phase, **fields):
        self.phase = phase
        self.emit(
            {
                "type": "state",
                "phase": phase,
                "command_id": self.active["command_id"] if self.active else None,
                **fields,
            }
        )

    def begin(self, command):
        if self.active is not None:
            raise ValueError("A command is already running")
        poses = self.session.poses()
        targets = {}
        for arm in ("left", "right"):
            data = command[arm]
            if data is not None:
                targets[arm] = local_target(
                    poses[arm], data["translation_m"], data["rotation_deg"]
                )
        self.active = command
        self.targets = targets
        self.start_grippers = {arm: self.session.gripper(arm) for arm in targets}
        self.stopping = False
        self.event("accepted", base_poses=poses, targets=targets)
        # Pure gripper commands hold arm drive targets and need no IK round trip.
        moving = {
            arm: target
            for arm, target in targets.items()
            if any(command[arm]["translation_m"]) or any(command[arm]["rotation_deg"])
        }
        try:
            requests = self.session.plan_requests(moving)
            self.future = self.executor.submit(self.session.plan, requests)
            self.event("planning")
        except Exception as exc:  # noqa: BLE001 -- surface backend setup failures to the operator
            self.finish("failed", error=str(exc))

    def hold(self):
        for arm in ("left", "right"):
            position = self.session.measured_joints(arm)
            self.session.drive(arm, position, np.zeros_like(position))

    def stop(self):
        if self.active is None:
            return
        self.hold()
        if self.phase == "planning":
            # GPU planning cannot be interrupted safely; discard its result.
            self.stopping = True
            self.event(
                "stopping",
                detail="Waiting for planner to finish; result will be discarded",
            )
        elif self.phase != "stopping":
            self.finish("stopped", actual_poses=self.session.poses())

    def finish(self, phase, **fields):
        self.event(phase, **fields)
        self.active = None
        self.future = None
        self.stopping = False

    def prepare_plans(self, plans):
        self.plans = {}
        for arm, result in plans.items():
            if result.get("status") != "Success":
                raise ValueError(
                    f"{arm} arm planning failed: {result.get('error', result.get('status'))}"
                )
            position = np.asarray(result.get("position"), dtype=float)
            velocity = np.asarray(result.get("velocity"), dtype=float)
            dof = len(self.session.measured_joints(arm))
            if (
                position.ndim != 2
                or position.shape[0] == 0
                or position.shape[1] != dof
                or velocity.shape != position.shape
                or not np.isfinite(position).all()
                or not np.isfinite(velocity).all()
            ):
                raise ValueError(f"{arm} planner returned an invalid trajectory")
            self.plans[arm] = (position, velocity)
        # Existing CuRobo interpolation_dt is 1/250 s. A shorter arm holds its
        # final waypoint while the other finishes; do not speed up either arm.
        self.duration = max(
            [0.2] + [(len(p) - 1) / 250 for p, _ in self.plans.values()]
        )
        self.tick = 0
        self.settle_tick = 0
        self.event("executing", duration_s=self.duration)

    def errors(self):
        poses = self.session.poses()
        errors = {
            arm: pose_error(target, poses[arm]) for arm, target in self.targets.items()
        }
        reached = all(
            error["position_m"] <= self.position_tolerance
            and error["rotation_deg"] <= self.rotation_tolerance
            for error in errors.values()
        )
        for arm in self.targets:
            target_gripper = self.active[arm]["gripper"]
            if target_gripper is not None:
                error = abs(self.session.measured_gripper(arm) - target_gripper)
                errors[arm]["gripper"] = error
                # A closing gripper can stop on a grasped object. Report the
                # aperture error without treating normal contact as IK failure.
        return poses, errors, reached

    def update(self):
        """Called only on the simulation thread, before one physics step."""
        if self.active is None:
            return
        try:
            if self.phase in {"planning", "stopping"}:
                if not self.future.done():
                    return
                if self.stopping:
                    self.finish("stopped", actual_poses=self.session.poses())
                    return
                self.prepare_plans(self.future.result())
            if self.phase == "executing":
                elapsed = self.tick * self.session.dt
                for arm, (positions, velocities) in self.plans.items():
                    index = min(int(elapsed * 250), len(positions) - 1)
                    velocity = (
                        velocities[index]
                        if elapsed < (len(positions) - 1) / 250
                        else np.zeros(positions.shape[1])
                    )
                    self.session.drive(arm, positions[index], velocity)
                fraction = min(1.0, elapsed / self.duration)
                for arm in self.targets:
                    target = self.active[arm]["gripper"]
                    if target is not None:
                        start = self.start_grippers[arm]
                        self.session.set_gripper(
                            arm, start + fraction * (target - start)
                        )
                self.tick += 1
                if elapsed >= self.duration:
                    self.event("settling")
            elif self.phase == "settling":
                self.settle_tick += 1
                poses, errors, reached = self.errors()
                elapsed = self.settle_tick * self.session.dt
                if (reached and elapsed >= 0.2) or elapsed >= 2.0:
                    self.finish(
                        "completed" if reached else "failed",
                        actual_poses=poses,
                        errors=errors,
                        targets=self.targets,
                        **(
                            {}
                            if reached
                            else {
                                "error": "Actual pose/gripper did not reach tolerance"
                            }
                        ),
                    )
        except Exception as exc:  # noqa: BLE001 -- third-party planners have heterogeneous exceptions
            if self.phase in {"executing", "settling"}:
                self.hold()
            self.finish("failed", error=str(exc), actual_poses=self.session.poses())

    @property
    def freeze_physics(self):
        return self.phase in {"planning", "stopping"} and self.active is not None

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
