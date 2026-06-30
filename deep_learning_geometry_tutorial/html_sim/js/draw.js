/* Canvas drawing helpers. */

const Draw = (() => {
  const world = { min: -1.1, max: 1.1 };

  function clear(ctx, canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function toCanvas(canvas, x, y) {
    const t = (x - world.min) / (world.max - world.min);
    const u = (y - world.min) / (world.max - world.min);
    return { x: t * canvas.width, y: (1 - u) * canvas.height };
  }

  function fromCanvas(canvas, px, py) {
    const x = world.min + (px / canvas.width) * (world.max - world.min);
    const y = world.max - (py / canvas.height) * (world.max - world.min);
    return { x, y };
  }

  function drawGrid(ctx, canvas) {
    ctx.save();
    ctx.strokeStyle = '#eff0f7';
    ctx.lineWidth = 1;
    for (let v = -1; v <= 1.0001; v += 0.25) {
      const a = toCanvas(canvas, v, world.min);
      const b = toCanvas(canvas, v, world.max);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();

      const c = toCanvas(canvas, world.min, v);
      const d = toCanvas(canvas, world.max, v);
      ctx.beginPath();
      ctx.moveTo(c.x, c.y);
      ctx.lineTo(d.x, d.y);
      ctx.stroke();
    }
    ctx.strokeStyle = '#b8bad2';
    ctx.lineWidth = 1.5;
    let a = toCanvas(canvas, 0, world.min);
    let b = toCanvas(canvas, 0, world.max);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    a = toCanvas(canvas, world.min, 0);
    b = toCanvas(canvas, world.max, 0);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.restore();
  }

  function heatColor(value, positiveColor = [255, 171, 76], negativeColor = [84, 126, 255]) {
    const t = Math.max(0, Math.min(1, value));
    const r = Math.round(negativeColor[0] * (1 - t) + positiveColor[0] * t);
    const g = Math.round(negativeColor[1] * (1 - t) + positiveColor[1] * t);
    const b = Math.round(negativeColor[2] * (1 - t) + positiveColor[2] * t);
    return `rgba(${r},${g},${b},0.42)`;
  }

  function drawScalarField(ctx, canvas, valueFn, resolution = 80) {
    const cellW = canvas.width / resolution;
    const cellH = canvas.height / resolution;
    for (let i = 0; i < resolution; i += 1) {
      for (let j = 0; j < resolution; j += 1) {
        const px = (i + 0.5) * cellW;
        const py = (j + 0.5) * cellH;
        const p = fromCanvas(canvas, px, py);
        const value = valueFn(p.x, p.y);
        ctx.fillStyle = heatColor(value);
        ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
      }
    }
  }

  function drawLineForPlane(ctx, canvas, w1, w2, b, color = '#222') {
    const points = [];
    const xs = [world.min, world.max];
    for (const x of xs) {
      if (Math.abs(w2) > 1e-9) {
        const y = -(w1 * x + b) / w2;
        if (y >= world.min && y <= world.max) points.push({ x, y });
      }
    }
    const ys = [world.min, world.max];
    for (const y of ys) {
      if (Math.abs(w1) > 1e-9) {
        const x = -(w2 * y + b) / w1;
        if (x >= world.min && x <= world.max) points.push({ x, y });
      }
    }
    if (points.length < 2) return;
    const p0 = toCanvas(canvas, points[0].x, points[0].y);
    const p1 = toCanvas(canvas, points[1].x, points[1].y);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.stroke();
    ctx.restore();
  }

  function drawTriangle(ctx, canvas, tri, fill = 'rgba(255, 190, 74, 0.35)', stroke = '#b76500') {
    const p0 = toCanvas(canvas, tri[0].x, tri[0].y);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < tri.length; i += 1) {
      const p = toCanvas(canvas, tri[i].x, tri[i].y);
      ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }

  function drawPolygon(ctx, canvas, poly, fill = 'rgba(230,230,240,0.2)', stroke = '#555') {
    const p0 = toCanvas(canvas, poly[0].x, poly[0].y);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < poly.length; i += 1) {
      const p = toCanvas(canvas, poly[i].x, poly[i].y);
      ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }

  function drawPoints(ctx, canvas, points, maxPoints = 1600) {
    ctx.save();
    const step = Math.max(1, Math.floor(points.length / maxPoints));
    for (let i = 0; i < points.length; i += step) {
      const p = points[i];
      const q = toCanvas(canvas, p.x, p.y);
      ctx.beginPath();
      ctx.arc(q.x, q.y, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = p.label ? 'rgba(220, 104, 0, 0.85)' : 'rgba(36, 72, 190, 0.70)';
      ctx.fill();
    }
    ctx.restore();
  }

  function drawLegend(ctx, canvas) {
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.90)';
    ctx.fillRect(14, 14, 210, 78);
    ctx.strokeStyle = '#ddddea';
    ctx.strokeRect(14, 14, 210, 78);
    ctx.fillStyle = 'rgba(220,104,0,0.85)';
    ctx.beginPath();
    ctx.arc(32, 38, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#20202a';
    ctx.font = '14px system-ui';
    ctx.fillText('inside triangular ray', 46, 43);
    ctx.fillStyle = 'rgba(36,72,190,0.70)';
    ctx.beginPath();
    ctx.arc(32, 68, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#20202a';
    ctx.fillText('outside ray', 46, 73);
    ctx.restore();
  }

  return {
    clear,
    drawGrid,
    drawScalarField,
    drawLineForPlane,
    drawTriangle,
    drawPolygon,
    drawPoints,
    drawLegend,
    toCanvas,
    fromCanvas,
    world,
  };
})();
