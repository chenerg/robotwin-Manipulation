// Run with: node tests/teleop_overlay_test.mjs
import assert from "node:assert/strict";
import {
  clipLine,
  originVisible,
  projectAxes,
  projectSegment,
} from "../teleop_web/static/overlay.js";

const camera = {
  width: 320,
  height: 240,
  near: 0.1,
  far: 100,
  intrinsic: [
    [100, 0, 160],
    [0, 100, 120],
    [0, 0, 1],
  ],
};
const axes = {
  origin: [0, 0, 1],
  basis: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};
const lines = projectAxes(axes, camera, 0.1);
assert.deepEqual(lines[0].start, [160, 120]);
assert.deepEqual(lines[0].end, [170, 120]);
assert.deepEqual(lines[1].end, [160, 130]);
assert.ok(
  Math.abs(lines[2].end[0] - 160) < 1e-9 &&
    Math.abs(lines[2].end[1] - 120) < 1e-9,
);
assert.equal(originVisible(axes, camera), true);
assert.equal(originVisible({ ...axes, origin: [0, 0, -1] }, camera), false);
assert.equal(projectSegment([0, 0, -1], [0, 0, -0.1], camera), null);
assert.equal(projectSegment([0, 0, 101], [0, 0, 102], camera), null);
assert.notEqual(projectSegment([0, 0, -1], [0.1, 0, 1], camera), null);
const partial = projectSegment([0, 0, 1], [5, 0, 1], camera);
assert.deepEqual(partial.end, [319, 120]);
assert.equal(partial.tipVisible, false);
assert.equal(clipLine([-2, -2], [-1, -1], 320, 240), null);
assert.deepEqual(clipLine([-10, 120], [330, 120], 320, 240).start, [0, 120]);
console.log(
  "EE projection: camera axes, near/far clipping, viewport clipping passed",
);
