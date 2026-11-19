// Camera data is in OpenCV axes: +X right, +Y down, +Z forward.
export const axisColors = ["#ff5f69", "#57e397", "#65a9ff"];

function interpolate(a, b, t) {
  return a.map((v, i) => v + t * (b[i] - v));
}

export function projectPoint(point, intrinsic) {
  const pixel = intrinsic.map((row) =>
    row.reduce((sum, value, i) => sum + value * point[i], 0),
  );
  return [pixel[0] / pixel[2], pixel[1] / pixel[2]];
}

export function clipLine(a, b, width, height) {
  const delta = [b[0] - a[0], b[1] - a[1]];
  let first = 0,
    last = 1;
  const constraints = [
    [-delta[0], a[0]],
    [delta[0], width - 1 - a[0]],
    [-delta[1], a[1]],
    [delta[1], height - 1 - a[1]],
  ];
  for (const [p, q] of constraints) {
    if (Math.abs(p) < 1e-12) {
      if (q < 0) return null;
    } else if (p < 0) first = Math.max(first, q / p);
    else last = Math.min(last, q / p);
    if (first > last) return null;
  }
  return {
    start: interpolate(a, b, first),
    end: interpolate(a, b, last),
    clipped: first > 0 || last < 1,
  };
}

export function projectSegment(origin, end, camera) {
  const near = camera.near ?? 0.1,
    far = camera.far ?? 100;
  const dz = end[2] - origin[2];
  let first = 0,
    last = 1;
  if (Math.abs(dz) < 1e-12) {
    if (origin[2] < near || origin[2] > far) return null;
  } else {
    const bounds = [(near - origin[2]) / dz, (far - origin[2]) / dz].sort(
      (a, b) => a - b,
    );
    first = Math.max(0, bounds[0]);
    last = Math.min(1, bounds[1]);
    if (first > last) return null;
  }
  const a = projectPoint(interpolate(origin, end, first), camera.intrinsic);
  const b = projectPoint(interpolate(origin, end, last), camera.intrinsic);
  if (![...a, ...b].every(Number.isFinite)) return null;
  const line = clipLine(a, b, camera.width, camera.height);
  if (!line) return null;
  line.tipVisible =
    last === 1 &&
    end[2] >= near &&
    b[0] >= 0 &&
    b[0] < camera.width &&
    b[1] >= 0 &&
    b[1] < camera.height;
  return line;
}

export function originVisible(axes, camera) {
  if (axes.origin[2] < camera.near || axes.origin[2] > camera.far) return false;
  const [x, y] = projectPoint(axes.origin, camera.intrinsic);
  return x >= 0 && x < camera.width && y >= 0 && y < camera.height;
}

export function projectAxes(axes, camera, length) {
  return [0, 1, 2].map((axis) => {
    const end = axes.origin.map((v, i) => v + length * axes.basis[i][axis]);
    return projectSegment(axes.origin, end, camera);
  });
}

function arrow(context, start, end, color, label, tip = true) {
  const [x, y] = end,
    angle = Math.atan2(y - start[1], x - start[0]);
  context.strokeStyle = color;
  context.fillStyle = color;
  context.beginPath();
  context.moveTo(...start);
  context.lineTo(x, y);
  context.stroke();
  if (tip) {
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(
      x - 6 * Math.cos(angle - 0.5),
      y - 6 * Math.sin(angle - 0.5),
    );
    context.lineTo(
      x - 6 * Math.cos(angle + 0.5),
      y - 6 * Math.sin(angle + 0.5),
    );
    context.closePath();
    context.fill();
    context.strokeStyle = "#09111d";
    context.lineWidth = 3;
    const tx = Math.max(2, Math.min(context.canvas.width - 30, x + 5));
    const ty = Math.max(13, Math.min(context.canvas.height - 4, y - 5));
    context.strokeText(label, tx, ty);
    context.fillText(label, tx, ty);
    context.lineWidth = 2;
  }
}

export function drawAxes(context, camera, options) {
  context.save();
  context.font = "bold 12px system-ui";
  context.lineWidth = 2;
  let inset = 0;
  for (const [arm, axes] of Object.entries(camera.axes)) {
    if (!options[arm]) continue;
    const prefix = arm === "left" ? "L" : "R";
    projectAxes(axes, camera, options.length).forEach((segment, axis) => {
      if (segment)
        arrow(
          context,
          segment.start,
          segment.end,
          axisColors[axis],
          `${prefix}:${"XYZ"[axis]}`,
          segment.tipVisible,
        );
    });
    if (!originVisible(axes, camera)) {
      const x = 14 + inset++ * 132,
        y = camera.height - 100;
      context.fillStyle = "#08121de0";
      context.fillRect(x, y, 124, 91);
      context.fillStyle = "#dce9f8";
      context.font = "11px system-ui";
      context.fillText(`${prefix} · EE 轴向（非原点）`, x + 5, y + 15);
      const start = [x + 61, y + 58];
      [0, 1, 2].forEach((axis) => {
        const vx = axes.basis[0][axis],
          vy = axes.basis[1][axis],
          vz = axes.basis[2][axis];
        if (Math.hypot(vx, vy) < 0.15) {
          context.fillStyle = axisColors[axis];
          context.fillText(
            `${"XYZ"[axis]}${vz > 0 ? "⊗" : "⊙"}`,
            start[0] + 4,
            start[1] + 13,
          );
        } else
          arrow(
            context,
            start,
            [start[0] + vx * 28, start[1] + vy * 28],
            axisColors[axis],
            "XYZ"[axis],
          );
      });
    }
  }
  context.restore();
}
