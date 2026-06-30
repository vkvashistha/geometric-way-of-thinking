/* Geometry helpers for the sun-ray classification task. */

const Geo = (() => {
  function triangleAreaSign(p, a, b) {
    return (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
  }

  function pointInTriangle(p, tri) {
    const [a, b, c] = tri;
    const d1 = triangleAreaSign(p, a, b);
    const d2 = triangleAreaSign(p, b, c);
    const d3 = triangleAreaSign(p, c, a);
    const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
    const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
    return !(hasNeg && hasPos);
  }

  function makeSunRays(rayCount, innerRadius = 0.48, outerRadius = 0.88) {
    const rays = [];
    const baseWidth = Math.min(0.38, 2 * innerRadius * Math.sin(Math.PI / rayCount) * 0.95);
    for (let i = 0; i < rayCount; i += 1) {
      const theta = (2 * Math.PI * i) / rayCount - Math.PI / 2;
      const dir = { x: Math.cos(theta), y: Math.sin(theta) };
      const tangent = { x: -dir.y, y: dir.x };
      const baseCenter = { x: innerRadius * dir.x, y: innerRadius * dir.y };
      const apex = { x: outerRadius * dir.x, y: outerRadius * dir.y };
      const half = baseWidth / 2;
      const p1 = { x: baseCenter.x + tangent.x * half, y: baseCenter.y + tangent.y * half };
      const p2 = apex;
      const p3 = { x: baseCenter.x - tangent.x * half, y: baseCenter.y - tangent.y * half };
      rays.push([p1, p2, p3]);
    }
    return rays;
  }

  function labelPoint(p, rays) {
    for (const tri of rays) {
      if (pointInTriangle(p, tri)) return 1;
    }
    return 0;
  }

  function randomPoint() {
    return {
      x: -1.05 + Math.random() * 2.10,
      y: -1.05 + Math.random() * 2.10,
    };
  }

  function shuffleInPlace(array) {
    for (let i = array.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  function sampleDataset(count, rays) {
    // Balanced sampling keeps the demo educational. With uniform sampling,
    // the ray region is small and a model can get high accuracy by predicting
    // "outside" almost everywhere.
    const points = [];
    const targetPos = Math.floor(count / 2);
    const targetNeg = count - targetPos;
    let pos = 0;
    let neg = 0;
    let attempts = 0;
    const maxAttempts = count * 200;

    while ((pos < targetPos || neg < targetNeg) && attempts < maxAttempts) {
      attempts += 1;
      const p = randomPoint();
      const label = labelPoint(p, rays);
      if (label === 1 && pos < targetPos) {
        points.push({ x: p.x, y: p.y, label });
        pos += 1;
      } else if (label === 0 && neg < targetNeg) {
        points.push({ x: p.x, y: p.y, label });
        neg += 1;
      }
    }

    while (points.length < count) {
      const p = randomPoint();
      points.push({ x: p.x, y: p.y, label: labelPoint(p, rays) });
    }

    shuffleInPlace(points);
    return points;
  }

  function squareVertices(halfSide = 0.34) {
    return [
      { x: -halfSide, y: -halfSide },
      { x: halfSide, y: -halfSide },
      { x: halfSide, y: halfSide },
      { x: -halfSide, y: halfSide },
    ];
  }

  function inwardHalfPlanesForTriangle(tri) {
    const centroid = {
      x: (tri[0].x + tri[1].x + tri[2].x) / 3,
      y: (tri[0].y + tri[1].y + tri[2].y) / 3,
    };
    const planes = [];
    for (let i = 0; i < 3; i += 1) {
      const a = tri[i];
      const b = tri[(i + 1) % 3];
      let w1 = b.y - a.y;
      let w2 = -(b.x - a.x);
      let bias = -(w1 * a.x + w2 * a.y);
      const insideValue = w1 * centroid.x + w2 * centroid.y + bias;
      if (insideValue < 0) {
        w1 *= -1;
        w2 *= -1;
        bias *= -1;
      }
      planes.push({ w1, w2, b: bias });
    }
    return planes;
  }

  function planeValue(plane, x, y) {
    return plane.w1 * x + plane.w2 * y + plane.b;
  }

  return {
    pointInTriangle,
    makeSunRays,
    labelPoint,
    sampleDataset,
    squareVertices,
    inwardHalfPlanesForTriangle,
    planeValue,
  };
})();
