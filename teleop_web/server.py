"""HTTP + separate control/frame WebSockets, with an isolated scene process."""

import asyncio
import copy
import json
import mimetypes
import multiprocessing as mp
import os
import queue
import signal
import struct
import time
import uuid
from contextlib import suppress
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlsplit

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

from .protocol import BUSY_PHASES, validate_move
from .session import catalog, normalize_embodiment, validate_scene
from .worker import run_worker

STATIC = Path(__file__).with_name("static")


class TeleopServer:
    def __init__(self, settings):
        self.settings = dict(settings)
        self.context = mp.get_context("spawn")
        self.process = None
        self.commands = self.events = self.frames = None
        self.scene_id = None
        self.phase = "starting"
        self.last_event = {}
        self.last_observation = None
        self.owner = None
        self.controls = {}
        self.seen_commands = set()
        self.history = {}
        self.history_entry = None
        self.packet = None
        self.frame_version = 0
        self.frame_changed = asyncio.Condition()
        self.resetting = False
        self.reset_task = None

    def snapshot(self, include_history=False):
        result = {
            "type": "snapshot",
            "scene_id": self.scene_id,
            "phase": self.phase,
            "busy": self.phase in BUSY_PHASES,
            "settings": self.settings,
            "owner": self.owner,
            "last_event": self.last_event,
            "observation": self.last_observation,
            "pose_frame": "tcp",
            "history_entry": self.history_entry,
        }
        if include_history:
            result["history"] = list(self.history.values())
        return result

    def record_event(self, event):
        """Keep one in-memory row per admitted command for this scene only."""
        command_id = event.get("command_id")
        if command_id in self.history:
            entry = {**self.history[command_id], "updated_at": time.time()}
            entry["phase"] = (
                "rejected" if event["type"] == "rejected" else event["phase"]
            )
            for key in ("base_poses", "targets", "actual_poses", "errors", "error"):
                if key in event:
                    entry[key] = event[key]
            self.history[command_id] = self.history_entry = entry
        elif event.get("phase") == "error":
            # Preserve an outcome even when the simulator exits mid-command.
            for entry in list(self.history.values()):
                if entry["phase"] in BUSY_PHASES:
                    self.record_event(
                        {
                            "type": "state",
                            "command_id": entry["command_id"],
                            "phase": "failed",
                            "error": event.get("error", "Simulator unavailable"),
                        }
                    )

    def broadcast(self, message):
        for channel in list(self.controls.values()):
            try:
                channel.put_nowait(message)
            except asyncio.QueueFull:
                # Disconnect a client that cannot consume command results.
                while not channel.empty():
                    channel.get_nowait()
                channel.put_nowait({"type": "overflow"})

    def start_worker(self):
        self.scene_id = uuid.uuid4().hex
        self.phase = "starting"
        self.last_event = {}
        self.last_observation = None
        self.packet = None
        self.seen_commands.clear()
        self.history.clear()
        self.history_entry = None
        self.commands = self.context.Queue(maxsize=32)
        self.events = self.context.Queue(maxsize=128)
        self.frames = self.context.Queue(maxsize=1)
        self.process = self.context.Process(
            target=run_worker,
            args=(
                self.settings,
                self.scene_id,
                self.commands,
                self.events,
                self.frames,
            ),
            name="robotwin-teleop",
        )
        self.process.start()

    def stop_worker(self):
        process = self.process
        if process is None:
            return
        with suppress(queue.Full, ValueError):
            self.commands.put_nowait({"type": "shutdown"})
        process.join(timeout=1.5)
        # Workers own their process group, including optional CuRobo children.
        owns_group = False
        if os.name == "posix":
            with suppress(ProcessLookupError):
                owns_group = os.getpgid(process.pid) == process.pid
        if owns_group:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        elif process.is_alive():
            process.terminate()
        process.join(timeout=1.5)
        if owns_group:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        elif process.is_alive():
            process.kill()
        process.join(timeout=1)
        for channel in (self.commands, self.events, self.frames):
            channel.cancel_join_thread()
            channel.close()
        self.process = None

    async def reset(self, settings):
        self.resetting = True
        self.phase = "resetting"
        self.broadcast(self.snapshot())
        try:
            await asyncio.to_thread(self.stop_worker)
            self.settings = settings
            self.start_worker()
        except Exception as exc:  # noqa: BLE001 -- keep the web service usable after scene startup errors
            self.phase = "error"
            self.last_event = {"error": str(exc)}
        finally:
            self.resetting = False
            self.broadcast(self.snapshot())

    async def pump(self):
        while True:
            if self.process is not None and not self.resetting:
                for _ in range(128):
                    try:
                        event = self.events.get_nowait()
                    except queue.Empty:
                        break
                    if event.get("scene_id") != self.scene_id:
                        continue
                    self.record_event(event)
                    if event["type"] == "state":
                        self.phase = event["phase"]
                        if event.get("command_id") and event[
                            "command_id"
                        ] == self.last_event.get("command_id"):
                            self.last_event = {**self.last_event, **event}
                        else:
                            self.last_event = event
                    elif event["type"] == "rejected":
                        self.phase = event["phase"]
                        self.last_event = event
                    self.broadcast(self.snapshot())
                try:
                    packet = self.frames.get_nowait()
                except queue.Empty:
                    packet = None
                if packet:
                    length = struct.unpack("!I", packet[:4])[0]
                    meta = json.loads(packet[4 : 4 + length])
                    if meta["scene_id"] == self.scene_id:
                        self.last_observation = {
                            key: meta[key]
                            for key in (
                                "poses",
                                "grippers",
                                "gripper_positions",
                                "task_success",
                                "sim_time",
                                "frame_id",
                            )
                        }
                        async with self.frame_changed:
                            self.packet = packet
                            self.frame_version += 1
                            self.frame_changed.notify_all()
                if not self.process.is_alive() and self.phase != "error":
                    # Allow the multiprocessing event feeder to deliver a final
                    # error before falling back to the exit code on the next poll.
                    if getattr(self, "dead_seen", None) == self.scene_id:
                        self.phase = "error"
                        self.last_event = {
                            "error": f"Simulator exited (code {self.process.exitcode})"
                        }
                        self.record_event(
                            {"type": "state", "phase": "error", **self.last_event}
                        )
                        self.broadcast(self.snapshot())
                    self.dead_seen = self.scene_id
            await asyncio.sleep(0.02)

    def dispatch(self, client_id, message):
        if not isinstance(message, dict):
            raise TypeError("Command must be a JSON object")
        kind = message.get("type")
        if kind == "claim":
            if self.owner not in (None, client_id):
                raise ValueError("Another browser controls this scene")
            self.owner = client_id
            self.broadcast(self.snapshot())
            return
        if self.owner != client_id:
            raise ValueError("Acquire control before operating the scene")
        if self.resetting:
            raise ValueError("Scene is being rebuilt")
        if message.get("scene_id") != self.scene_id:
            raise ValueError("Scene changed; use the current scene_id")
        if kind == "reset":
            settings = dict(self.settings)
            for key in (
                "task_name",
                "task_config",
                "seed",
                "embodiment",
                "head_camera_type",
                "wrist_camera_type",
            ):
                settings[key] = message.get(key, settings.get(key))
            settings["embodiment"] = normalize_embodiment(settings["embodiment"])
            validate_scene(
                settings["task_name"],
                settings["task_config"],
                settings["seed"],
                settings["embodiment"],
                settings["head_camera_type"],
                settings["wrist_camera_type"],
            )
            # Claim busy state before yielding to the reset task.
            self.resetting = True
            self.phase = "resetting"
            self.broadcast(self.snapshot())
            self.reset_task = asyncio.create_task(self.reset(settings))
            return
        if kind == "stop":
            if self.process is not None and self.process.is_alive():
                self.commands.put_nowait({"type": "stop", "scene_id": self.scene_id})
            return
        if kind != "move":
            raise ValueError("Unknown command type")
        command = validate_move(message)
        if command["command_id"] in self.seen_commands:
            raise ValueError("Duplicate command_id; movement was not repeated")
        if self.phase in BUSY_PHASES or self.phase == "error":
            raise ValueError(f"Scene is not ready ({self.phase})")
        if len(self.seen_commands) >= 10000:
            raise ValueError("Reset the scene before sending more commands")
        self.commands.put_nowait(command)
        self.seen_commands.add(command["command_id"])
        now = time.time()
        entry = {
            **copy.deepcopy(command),
            "sequence": len(self.history) + 1,
            "accepted_at": now,
            "updated_at": now,
            "phase": "accepted",
        }
        self.history[command["command_id"]] = self.history_entry = entry
        self.phase = "accepted"
        self.last_event = {"command_id": command["command_id"], "phase": "accepted"}
        self.broadcast(self.snapshot())

    async def control(self, connection):
        client_id = str(connection.id)
        channel = asyncio.Queue(maxsize=64)
        self.controls[client_id] = channel
        await connection.send(
            json.dumps(
                {
                    "type": "hello",
                    "client_id": client_id,
                    "catalog": catalog(),
                    "state": self.snapshot(include_history=True),
                }
            )
        )

        async def send_events():
            while True:
                message = await channel.get()
                if message["type"] == "overflow":
                    await connection.close(1013, "Status consumer too slow; reconnect")
                    return
                await connection.send(json.dumps(message, allow_nan=False))

        sender = asyncio.create_task(send_events())
        try:
            async for raw in connection:
                message = None
                try:
                    if not isinstance(raw, str):
                        raise TypeError("Control messages must be JSON text")
                    message = json.loads(raw)
                    self.dispatch(client_id, message)
                except (ValueError, TypeError, queue.Full) as exc:
                    try:
                        channel.put_nowait(
                            {
                                "type": "rejected",
                                "error": str(exc) or "Command queue is full",
                                "command_id": message.get("command_id")
                                if isinstance(message, dict)
                                else None,
                            }
                        )
                    except asyncio.QueueFull:
                        await connection.close(
                            1013, "Status consumer too slow; reconnect"
                        )
        finally:
            sender.cancel()
            with suppress(asyncio.CancelledError, ConnectionClosed):
                await sender
            self.controls.pop(client_id, None)
            if self.owner == client_id:
                if (
                    self.process is not None
                    and self.process.is_alive()
                    and not self.resetting
                ):
                    with suppress(queue.Full):
                        self.commands.put_nowait(
                            {"type": "stop", "scene_id": self.scene_id}
                        )
                self.owner = None
                self.broadcast(self.snapshot())

    async def media(self, connection):
        async def send_frames():
            version = -1
            while True:
                async with self.frame_changed:
                    await self.frame_changed.wait_for(
                        lambda seen=version: (
                            self.packet is not None and self.frame_version != seen
                        )
                    )
                    packet, version = self.packet, self.frame_version
                try:
                    await asyncio.wait_for(connection.send(packet), timeout=2)
                except asyncio.TimeoutError:
                    await connection.close(1013, "Frame consumer too slow; reconnect")
                    return

        sender = asyncio.create_task(send_frames())
        closed = asyncio.create_task(connection.wait_closed())
        try:
            done, _ = await asyncio.wait(
                (sender, closed), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        finally:
            sender.cancel()
            closed.cancel()
            with suppress(asyncio.CancelledError, ConnectionClosed):
                await sender
            with suppress(asyncio.CancelledError):
                await closed

    async def handler(self, connection):
        try:
            if urlsplit(connection.request.path).path == "/ws/control":
                await self.control(connection)
            else:
                await self.media(connection)
        except ConnectionClosed:
            pass

    def http(self, connection, request):
        path = urlsplit(request.path).path
        if path in {"/ws/control", "/ws/frames"}:
            origin = request.headers.get("Origin")
            if origin and urlsplit(origin).netloc != request.headers.get("Host"):
                return connection.respond(
                    HTTPStatus.FORBIDDEN, "Origin does not match this server"
                )
            return None
        if path == "/api/state":
            body, content_type = (
                json.dumps(self.snapshot(include_history=True)).encode(),
                "application/json",
            )
        elif path == "/api/catalog":
            body, content_type = json.dumps(catalog()).encode(), "application/json"
        elif path in {"/", "/index.html", "/app.js", "/overlay.js", "/style.css"}:
            file = STATIC / ("index.html" if path == "/" else path[1:])
            body = file.read_bytes()
            content_type = (
                mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            )
        else:
            return connection.respond(HTTPStatus.NOT_FOUND, "Not found")
        headers = Headers(
            {
                "Content-Type": content_type + "; charset=utf-8",
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
                "Connection": "close",
                "X-Content-Type-Options": "nosniff",
            }
        )
        return Response(200, "OK", headers, body)


async def run_server(settings, host, port):
    app = TeleopServer(settings)
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signum, shutdown.set)
    pump = None
    try:
        async with serve(
            app.handler,
            host,
            port,
            process_request=app.http,
            compression=None,
            max_size=16384,
            max_queue=8,
            close_timeout=2,
        ):
            app.start_worker()
            pump = asyncio.create_task(app.pump())
            print(f"RoboTwin teleop: http://{host}:{port}", flush=True)
            await shutdown.wait()
    finally:
        if app.reset_task is not None:
            with suppress(Exception):
                await app.reset_task
        if pump is not None:
            pump.cancel()
            with suppress(asyncio.CancelledError):
                await pump
        await asyncio.to_thread(app.stop_worker)
