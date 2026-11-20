import { drawAxes } from "/overlay.js";

const $ = (id) => document.getElementById(id);
const arms = ["left", "right"];
const names = { left: "左", right: "右" };
const phaseNames = {
  starting: "仿真初始化中",
  resetting: "场景重建中",
  ready: "等待输入",
  accepted: "已接受指令",
  planning: "IK / 轨迹规划中",
  executing: "运动执行中",
  settling: "等待到达目标",
  stopping: "停止中，等待规划结束",
  completed: "运动已完成",
  failed: "运动未完成",
  stopped: "运动已停止",
  rejected: "指令被拒绝",
  error: "仿真不可用",
};
let state = null,
  clientId = null,
  controlSocket = null,
  mediaSocket = null;
let frame = null,
  pendingPacket = null,
  decoding = false,
  reconnectTimer = null;
let lastFrameTime = 0,
  fps = 0,
  lastLogKey = "";
let pendingCommand = null;
const historyRows = new Map();
const deltaFields = ["x", "y", "z", "roll", "pitch", "yaw"];

for (const arm of arms) {
  const fields = [
    ["x", "Δx", "cm"],
    ["y", "Δy", "cm"],
    ["z", "Δz", "cm"],
    ["roll", "Δroll", "°"],
    ["pitch", "Δpitch", "°"],
    ["yaw", "Δyaw", "°"],
  ];
  document.querySelector(`[data-arm="${arm}"]`).innerHTML = `
    <div class="arm-heading"><h2 id="${arm}-title">${names[arm]}夹爪</h2><span>${arm.toUpperCase()} / TCP LOCAL</span></div>
    <div class="fields">${fields.map(([key, label, unit]) => `<label for="${arm}-${key}">${label} <span class="unit">${unit}</span><input id="${arm}-${key}" type="number" step="any" value="0"></label>`).join("")}</div>
    <div class="gripper"><label><input id="${arm}-gripper-enabled" type="checkbox">同时设置夹爪开合<input id="${arm}-gripper" aria-label="${names[arm]}夹爪开合值" type="number" min="0" max="1" step="any" value="1"></label></div>
    <div class="arm-actions"><button id="${arm}-execute" class="primary">执行${names[arm]}臂</button><button id="${arm}-zero">清零增量</button></div>
    <div id="${arm}-pose" class="pose">等待实际位姿</div>
    <div id="${arm}-target" class="pose">目标：尚未发送</div>`;
  $(`${arm}-execute`).onclick = () => move([arm]);
  $(`${arm}-zero`).onclick = () =>
    fields.forEach(([key]) => {
      $(`${arm}-${key}`).value = 0;
    });
}

function feedback(text, error = false) {
  $("feedback").textContent = text;
  $("feedback").classList.toggle("error", error);
}
function log(text) {
  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  $("log").prepend(item);
  while ($("log").children.length > 100) $("log").lastChild.remove();
}
function connected() {
  return controlSocket?.readyState === WebSocket.OPEN;
}
function send(message) {
  if (!connected()) {
    feedback("连接已断开，请等待重连。", true);
    return false;
  }
  controlSocket.send(JSON.stringify({ ...message, scene_id: state?.scene_id }));
  return true;
}
function number(id) {
  const input = $(id),
    value = input.valueAsNumber;
  if (!input.checkValidity() || !Number.isFinite(value))
    throw new Error(`请输入有效数值：${input.labels?.[0]?.textContent || id}`);
  return value;
}
function move(selected) {
  if (pendingCommand || state?.busy) return;
  try {
    const message = {
      type: "move",
      command_id:
        globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
      frame: "tcp_local",
      left: null,
      right: null,
    };
    for (const arm of selected)
      message[arm] = {
        translation_m: ["x", "y", "z"].map(
          (axis) => number(`${arm}-${axis}`) / 100,
        ),
        rotation_deg: Object.fromEntries(
          ["roll", "pitch", "yaw"].map((axis) => [
            axis,
            number(`${arm}-${axis}`),
          ]),
        ),
        gripper: $(`${arm}-gripper-enabled`).checked
          ? number(`${arm}-gripper`)
          : null,
      };
    if (send(message)) {
      pendingCommand = {
        id: message.command_id,
        arms: selected,
        sceneId: state.scene_id,
      };
      updateControls();
    }
  } catch (error) {
    feedback(error.message, true);
  }
}
function displayPose(pose) {
  if (!pose) return "等待位姿";
  return `xyz [cm] ${pose
    .slice(0, 3)
    .map((v) => (v * 100).toFixed(2))
    .join("  ")}\nwxyz     ${pose
    .slice(3)
    .map((v) => v.toFixed(4))
    .join("  ")}`;
}
function updateControls() {
  const owns = state?.owner === clientId && connected();
  const ready =
    owns && state && !state.busy && !pendingCommand && state.phase !== "error";
  arms.forEach((arm) => {
    $(`${arm}-execute`).disabled = !ready;
    $(`${arm}-zero`).disabled = !ready;
    for (const field of [...deltaFields, "gripper", "gripper-enabled"])
      $(`${arm}-${field}`).disabled = !ready;
  });
  $("both").disabled = !ready;
  $("stop").disabled =
    !owns || !state?.busy || ["starting", "resetting"].includes(state?.phase);
  $("reset").disabled = !owns || state?.phase === "resetting";
  $("claim").disabled =
    !connected() || (state?.owner != null && state.owner !== clientId) || owns;
  $("claim").textContent = owns
    ? "已获得控制权"
    : state?.owner
      ? "其他页面正在控制"
      : "获取控制权";
}
function updateEmbodimentFields() {
  const mode = $("embodiment-mode").value;
  $("embodiment-left-field").hidden = !mode;
  $("embodiment-right-field").hidden = mode !== "pair";
  $("embodiment-spacing-field").hidden = mode !== "pair";
}
function selectedEmbodiment() {
  const mode = $("embodiment-mode").value;
  if (!mode) return null;
  if (mode === "single") return [$("embodiment-left").value];
  return [
    $("embodiment-left").value,
    $("embodiment-right").value,
    number("embodiment-spacing") / 100,
  ];
}
function applyHistoryEntry(entry) {
  if (!entry) return;
  let row = historyRows.get(entry.command_id);
  if (!row) {
    const item = document.createElement("li");
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const body = document.createElement("pre");
    details.append(summary, body);
    item.append(details);
    $("command-history").prepend(item);
    row = { summary, body };
    historyRows.set(entry.command_id, row);
  }
  const selected = arms.filter((arm) => entry[arm]);
  row.summary.textContent = `#${entry.sequence} · ${new Date(entry.accepted_at * 1000).toLocaleTimeString()} · ${selected.map((arm) => names[arm] + "臂").join(" + ")} · ${phaseNames[entry.phase] || entry.phase}`;
  const lines = [`命令 ID：${entry.command_id}`];
  for (const arm of selected) {
    const command = entry[arm];
    lines.push(
      `${names[arm]}臂 Δxyz [cm]：${command.translation_m.map((v) => +(v * 100).toFixed(6)).join(" / ")}`,
    );
    lines.push(
      `Δroll/pitch/yaw [°]：${command.rotation_deg.join(" / ")}；夹爪目标：${command.gripper ?? "保留"}`,
    );
    for (const [key, label] of [
      ["base_poses", "起始"],
      ["targets", "目标"],
      ["actual_poses", "结束实际"],
    ])
      if (entry[key]?.[arm])
        lines.push(`${label}（世界系）\n${displayPose(entry[key][arm])}`);
    const error = entry.errors?.[arm];
    if (error)
      lines.push(
        `到达误差：${(error.position_m * 100).toFixed(3)} cm / ${error.rotation_deg.toFixed(2)}°${error.gripper == null ? "" : `；夹爪开合误差 ${error.gripper.toFixed(3)}`}`,
      );
  }
  if (entry.error) lines.push(`错误：${entry.error}`);
  row.body.textContent = lines.join("\n");
  if (
    pendingCommand?.id === entry.command_id &&
    ["completed", "failed", "stopped", "rejected"].includes(entry.phase)
  ) {
    if (entry.phase !== "rejected")
      for (const arm of pendingCommand.arms)
        for (const field of deltaFields) $(`${arm}-${field}`).value = 0;
    pendingCommand = null;
  }
}
function clearFrame() {
  if (frame) frame.bitmaps.forEach((bitmap) => bitmap.close());
  frame = null;
  pendingPacket = null;
  lastFrameTime = 0;
  for (const name of ["head_camera", "left_camera", "right_camera"]) {
    const canvas = $(name);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    canvas.parentElement.classList.remove("has-frame");
  }
  $("frame-info").textContent = "等待新场景图像";
  arms.forEach((arm) => {
    $(`${arm}-pose`).textContent = "等待实际位姿";
    $(`${arm}-target`).textContent = "目标：尚未发送";
  });
}
function applyState(next) {
  if (state?.scene_id !== next.scene_id) {
    clearFrame();
    historyRows.clear();
    $("command-history").replaceChildren();
    $("log").replaceChildren();
    if (pendingCommand?.sceneId !== next.scene_id) pendingCommand = null;
    lastLogKey = "";
    $("task").value = next.settings.task_name;
    $("config").value = next.settings.task_config;
    $("seed").value = next.settings.seed;
    const embodiment = next.settings.embodiment;
    $("embodiment-mode").value = !embodiment
      ? ""
      : embodiment.length === 1
        ? "single"
        : "pair";
    if (embodiment) $("embodiment-left").value = embodiment[0];
    if (embodiment?.length === 3) {
      $("embodiment-right").value = embodiment[1];
      $("embodiment-spacing").value = embodiment[2] * 100;
    }
    $("head-camera-type").value = next.settings.head_camera_type || "";
    $("wrist-camera-type").value = next.settings.wrist_camera_type || "";
    updateEmbodimentFields();
  }
  state = next;
  if (next.history) {
    for (const entry of next.history) applyHistoryEntry(entry);
    if (
      pendingCommand &&
      !next.history.some((entry) => entry.command_id === pendingCommand.id)
    )
      pendingCommand = null;
  }
  applyHistoryEntry(next.history_entry);
  $("phase").textContent = phaseNames[state.phase] || state.phase;
  const embodiment = state.settings.embodiment;
  const embodimentLabel = !embodiment
    ? "沿用配置"
    : embodiment.length === 1
      ? embodiment[0]
      : `${embodiment[0]} + ${embodiment[1]}（间距 ${+(embodiment[2] * 100).toFixed(4)} cm）`;
  $("scene-info").textContent =
    `${state.settings.task_name} / ${state.settings.task_config} / seed ${state.settings.seed} / 本体 ${embodimentLabel} / 主相机 ${state.settings.head_camera_type || "沿用配置"} / 腕部 ${state.settings.wrist_camera_type || "沿用配置"}`;
  const event = state.last_event || {};
  const key = JSON.stringify(event);
  if (key !== lastLogKey && Object.keys(event).length) {
    lastLogKey = key;
    let detail = phaseNames[event.phase] || "状态更新";
    if (event.error) detail += `：${event.error}`;
    if (event.errors)
      detail +=
        "；" +
        Object.entries(event.errors)
          .map(
            ([arm, error]) =>
              `${names[arm]}臂 ${(error.position_m * 1000).toFixed(2)} mm / ${error.rotation_deg.toFixed(2)}°${error.gripper == null ? "" : ` / 开合误差 ${error.gripper.toFixed(3)}`}`,
          )
          .join("；");
    feedback(detail, !!event.error);
    log(detail);
  }
  for (const arm of arms) {
    if (event.targets?.[arm])
      $(`${arm}-target`).textContent =
        `TCP 世界目标\n${displayPose(event.targets[arm])}`;
    if (event.actual_poses?.[arm])
      $(`${arm}-pose`).textContent =
        `实际 TCP（世界系）\n${displayPose(event.actual_poses[arm])}`;
  }
  updateControls();
}
function draw() {
  if (!frame) return;
  const length = $("axis-length").valueAsNumber;
  const options = {
    left: $("axis-left").checked,
    right: $("axis-right").checked,
    length: Number.isFinite(length) && length > 0 ? length / 100 : 0.05,
  };
  frame.meta.cameras.forEach((camera, index) => {
    const canvas = $(camera.name);
    canvas.width = camera.width;
    canvas.height = camera.height;
    const context = canvas.getContext("2d");
    context.drawImage(frame.bitmaps[index], 0, 0);
    drawAxes(context, camera, options);
    canvas.parentElement.classList.add("has-frame");
  });
}
async function decodeFrames() {
  if (decoding) return;
  decoding = true;
  try {
    while (pendingPacket) {
      const packet = pendingPacket;
      pendingPacket = null;
      const headerSize = new DataView(packet).getUint32(0);
      const meta = JSON.parse(
        new TextDecoder().decode(new Uint8Array(packet, 4, headerSize)),
      );
      if (meta.scene_id !== state?.scene_id) continue;
      let offset = 4 + headerSize;
      const images = meta.cameras.map((camera) => {
        const blob = new Blob(
          [new Uint8Array(packet, offset, camera.jpeg_bytes)],
          { type: "image/jpeg" },
        );
        offset += camera.jpeg_bytes;
        return createImageBitmap(blob);
      });
      const bitmaps = await Promise.all(images);
      if (
        meta.scene_id !== state?.scene_id ||
        (frame && meta.frame_id <= frame.meta.frame_id)
      ) {
        bitmaps.forEach((bitmap) => bitmap.close());
        continue;
      }
      if (frame) frame.bitmaps.forEach((bitmap) => bitmap.close());
      frame = { meta, bitmaps };
      draw();
      const now = performance.now();
      if (lastFrameTime) fps = 0.7 * fps + (0.3 * 1000) / (now - lastFrameTime);
      lastFrameTime = now;
      $("frame-info").textContent =
        `${fps.toFixed(1)} FPS · 帧 ${meta.frame_id} · 仿真 ${meta.sim_time.toFixed(2)} s`;
      $("task-success").textContent =
        `任务状态：${meta.task_success ? "已满足成功条件" : "进行中"}`;
      arms.forEach((arm) => {
        $(`${arm}-pose`).textContent =
          `实际 TCP（世界系）\n${displayPose(meta.poses[arm])}\n开合 实际 ${meta.gripper_positions[arm].toFixed(2)} / 目标 ${meta.grippers[arm].toFixed(2)}`;
      });
    }
  } catch (error) {
    feedback(`图像解码失败：${error.message}`, true);
  } finally {
    decoding = false;
  }
}
function openMedia() {
  if (
    !connected() ||
    [WebSocket.CONNECTING, WebSocket.OPEN].includes(mediaSocket?.readyState)
  )
    return;
  mediaSocket = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/frames`,
  );
  mediaSocket.binaryType = "arraybuffer";
  mediaSocket.onmessage = (event) => {
    pendingPacket = event.data;
    decodeFrames();
  };
  mediaSocket.onclose = () => {
    $("frame-info").textContent = "图像流断开，正在重连";
    if (connected()) setTimeout(openMedia, 1500);
  };
}
function connect() {
  controlSocket = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/control`,
  );
  controlSocket.onopen = () => {
    $("connection").textContent = "控制通道已连接";
    $("connection-dot").classList.add("online");
  };
  controlSocket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "hello") {
      clientId = message.client_id;
      for (const [id, entries] of [
        ["task", message.catalog.tasks],
        ["config", message.catalog.task_configs],
        ["embodiment-left", message.catalog.embodiments],
        ["embodiment-right", message.catalog.embodiments],
      ]) {
        $(id).replaceChildren(
          ...entries.map((value) => new Option(value, value)),
        );
      }
      for (const id of ["head-camera-type", "wrist-camera-type"])
        $(id).replaceChildren(
          new Option("沿用任务配置", ""),
          ...message.catalog.camera_types.map(
            (value) => new Option(value, value),
          ),
        );
      state = null;
      applyState(message.state);
      openMedia();
      send({ type: "claim" });
    } else if (message.type === "snapshot") applyState(message);
    else if (message.type === "rejected") {
      if (pendingCommand?.id === message.command_id) pendingCommand = null;
      updateControls();
      feedback(message.error, true);
      log(message.error);
    }
  };
  controlSocket.onclose = () => {
    $("connection").textContent = "连接断开，正在重连";
    $("connection-dot").classList.remove("online");
    mediaSocket?.close();
    updateControls();
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, 1500);
  };
}
$("both").onclick = () => move(arms);
$("stop").onclick = () => send({ type: "stop" });
$("claim").onclick = () => send({ type: "claim" });
$("reset").onclick = () => {
  try {
    send({
      type: "reset",
      task_name: $("task").value,
      task_config: $("config").value,
      seed: number("seed"),
      embodiment: selectedEmbodiment(),
      head_camera_type: $("head-camera-type").value || null,
      wrist_camera_type: $("wrist-camera-type").value || null,
    });
  } catch (error) {
    feedback(error.message, true);
  }
};
$("clear-log").onclick = () => $("log").replaceChildren();
$("embodiment-mode").onchange = updateEmbodimentFields;
for (const id of ["axis-left", "axis-right", "axis-length"])
  $(id).addEventListener("input", draw);
setInterval(() => {
  if (frame && performance.now() - lastFrameTime > 3000)
    $("frame-info").textContent = "画面已超过 3 秒未更新";
}, 1000);
updateControls();
connect();
