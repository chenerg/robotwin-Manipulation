// Exercise the actual app event handlers with a minimal DOM/WebSocket fixture.
// Run with Node; this verifies browser logic, not rendering or robot physics.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const elements = new Map();
class Element {
  constructor() {
    this.value = "0";
    this.checked = false;
    this.children = [];
    this.classList = { add() {}, remove() {}, toggle() {} };
    this.parentElement = { classList: this.classList };
  }
  get valueAsNumber() {
    return this.value === "" ? NaN : Number(this.value);
  }
  checkValidity() {
    return Number.isFinite(this.valueAsNumber);
  }
  set innerHTML(html) {
    for (const [, id] of html.matchAll(/id="([^"]+)"/g))
      elements.set(id, new Element());
  }
  append(...children) {
    this.children.push(...children);
  }
  prepend(...children) {
    this.children.unshift(...children);
  }
  replaceChildren(...children) {
    this.children = children;
  }
  addEventListener() {}
  getContext() {
    return { clearRect() {} };
  }
}
const html = readFileSync(
  new URL("../teleop_web/static/index.html", import.meta.url),
  "utf8",
);
for (const [, id] of html.matchAll(/id="([^"]+)"/g))
  elements.set(id, new Element());
const panels = { left: new Element(), right: new Element() };
const sockets = [];
class Socket {
  static OPEN = 1;
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    this.sent = [];
    sockets.push(this);
  }
  send(text) {
    this.sent.push(JSON.parse(text));
  }
  close() {
    this.readyState = 3;
  }
}
const context = vm.createContext({
  document: {
    getElementById: (id) => elements.get(id),
    createElement: () => new Element(),
    querySelector: (query) =>
      panels[query.includes('"left"') ? "left" : "right"],
  },
  WebSocket: Socket,
  Option: class extends Element {
    constructor(text, value) {
      super();
      this.textContent = text;
      this.value = value;
    }
  },
  location: { protocol: "http:", host: "localhost" },
  setInterval() {},
  setTimeout() {},
  clearTimeout() {},
  performance: { now: () => 0 },
  drawAxes() {},
  console,
});
const source = readFileSync(
  new URL("../teleop_web/static/app.js", import.meta.url),
  "utf8",
);
vm.runInContext(source.replace(/^import .*;\n/, ""), context);
const element = (id) => elements.get(id);
const control = sockets[0];
const settings = {
  task_name: "handover_block",
  task_config: "demo_clean",
  seed: 0,
  embodiment: ["piper", "franka-panda", 0.6],
  head_camera_type: "L515",
  wrist_camera_type: "Large_L515",
};
const snapshot = (fields = {}) => ({
  type: "snapshot",
  scene_id: "s1",
  phase: "ready",
  busy: false,
  owner: "browser",
  settings,
  last_event: {},
  history_entry: null,
  ...fields,
});
const receive = (message) =>
  control.onmessage({ data: JSON.stringify(message) });
receive({
  type: "hello",
  client_id: "browser",
  catalog: {
    tasks: ["handover_block"],
    task_configs: ["demo_clean"],
    embodiments: ["piper", "franka-panda"],
    camera_types: ["D435", "L515", "Large_L515"],
  },
  state: snapshot({ history: [] }),
});
assert.equal(element("embodiment-mode").value, "pair");
assert.equal(element("embodiment-spacing").value, 60);
assert.equal(element("head-camera-type").value, "L515");
assert.equal(element("embodiment-right-field").hidden, false);

const submit = (both = false) => {
  element("left-x").value = "2.5";
  element("left-roll").value = "0.5";
  element("left-gripper-enabled").checked = true;
  element("left-gripper").value = "0.4";
  element("right-x").value = "7";
  element(both ? "both" : "left-execute").onclick();
  const command = control.sent.at(-1);
  assert.equal(command.frame, "tcp_local");
  assert.equal(command.left.translation_m[0], 0.025);
  assert.equal(command.left.rotation_deg.roll, 0.5);
  assert.equal(command.left.gripper, 0.4);
  assert.equal(element("left-x").disabled, true);
  return command;
};
let sequence = 0;
const outcome = (command, phase) => ({
  ...command,
  sequence: ++sequence,
  accepted_at: 1,
  phase,
  left: { ...command.left, rotation_deg: [0.5, 0, 0] },
  right: command.right ? { ...command.right, rotation_deg: [0, 0, 0] } : null,
});
for (const phase of ["completed", "failed", "stopped"]) {
  const command = submit();
  const entry = outcome(command, phase);
  receive(snapshot({ phase, history_entry: entry }));
  assert.equal(element("left-x").value, 0);
  assert.equal(element("left-roll").value, 0);
  assert.equal(element("right-x").value, "7");
  assert.equal(element("left-gripper").value, "0.4");
  assert.equal(element("left-gripper-enabled").checked, true);
  assert.equal(element("left-x").disabled, false);
  const rowCount = element("command-history").children.length;
  receive(snapshot({ phase, history_entry: entry }));
  assert.equal(element("command-history").children.length, rowCount);
}
let command = submit(true);
receive(
  snapshot({
    phase: "completed",
    history_entry: outcome(command, "completed"),
  }),
);
assert.equal(element("right-x").value, 0);
command = submit();
receive({ type: "rejected", command_id: command.command_id, error: "busy" });
assert.equal(element("left-x").value, "2.5");
assert.equal(element("left-x").disabled, false);

// An unrelated reset error must not release a pending movement or clear inputs.
command = submit();
receive({ type: "rejected", command_id: null, error: "invalid reset" });
assert.equal(element("left-x").disabled, true);
receive(
  snapshot({
    phase: "completed",
    history_entry: outcome(command, "completed"),
  }),
);

element("embodiment-mode").value = "pair";
element("embodiment-left").value = "piper";
element("embodiment-right").value = "franka-panda";
element("embodiment-spacing").value = "60";
element("reset").onclick();
assert.deepEqual(control.sent.at(-1).embodiment, [
  "piper",
  "franka-panda",
  0.6,
]);
assert.equal(control.sent.at(-1).head_camera_type, "L515");
element("embodiment-mode").value = "";
element("embodiment-mode").onchange();
assert.equal(element("embodiment-left-field").hidden, true);
element("reset").onclick();
assert.equal(control.sent.at(-1).embodiment, null);

const restored = outcome(command, "completed");
receive(snapshot({ scene_id: "s2", history: [restored] }));
assert.equal(element("command-history").children.length, 1);
element("clear-log").onclick();
assert.equal(element("command-history").children.length, 1);
receive(snapshot({ scene_id: "s3" }));
assert.equal(element("command-history").children.length, 0);
console.log(
  "Teleop controls: cm conversion, terminal clearing, rejection, registry selection and episode history passed",
);
