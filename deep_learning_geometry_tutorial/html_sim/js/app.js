/* Main application logic. */

(() => {
  const canvas = document.getElementById('mainCanvas');
  const ctx = canvas.getContext('2d');

  const modeSelect = document.getElementById('modeSelect');
  const statusEl = document.getElementById('status');
  const explanationEl = document.getElementById('explanationText');

  const w1Slider = document.getElementById('w1Slider');
  const w2Slider = document.getElementById('w2Slider');
  const bSlider = document.getElementById('bSlider');
  const singleActivation = document.getElementById('singleActivation');
  const triangleShow = document.getElementById('triangleShow');
  const rayCountSelect = document.getElementById('rayCount');
  const sampleCountSelect = document.getElementById('sampleCount');
  const newSamplesBtn = document.getElementById('newSamplesBtn');
  const architectureSelect = document.getElementById('architectureSelect');
  const hiddenActivation = document.getElementById('hiddenActivation');
  const learningRate = document.getElementById('learningRate');
  const lrValue = document.getElementById('lrValue');
  const resetModelBtn = document.getElementById('resetModelBtn');
  const trainOneBtn = document.getElementById('trainOneBtn');
  const train100Btn = document.getElementById('train100Btn');
  const train1000Btn = document.getElementById('train1000Btn');
  const runCompareBtn = document.getElementById('runCompareBtn');
  const compareResults = document.getElementById('compareResults');

  const groups = {
    single: document.getElementById('singleControls'),
    triangle: document.getElementById('triangleControls'),
    sun: document.getElementById('sunControls'),
    train: document.getElementById('trainControls'),
    compare: document.getElementById('compareControls'),
  };

  let rays = Geo.makeSunRays(parseInt(rayCountSelect.value, 10));
  let dataset = Geo.sampleDataset(parseInt(sampleCountSelect.value, 10), rays);
  let model = makeModel();
  let lastMetrics = null;

  function makeModel() {
    const arch = NN.parseArchitecture(architectureSelect.value);
    return new NN.MLP(arch, hiddenActivation.value);
  }

  function regenerateDataset() {
    rays = Geo.makeSunRays(parseInt(rayCountSelect.value, 10));
    dataset = Geo.sampleDataset(parseInt(sampleCountSelect.value, 10), rays);
    model = makeModel();
    lastMetrics = null;
    compareResults.innerHTML = '';
    draw();
  }

  function updateControlVisibility() {
    const mode = modeSelect.value;
    groups.single.classList.toggle('hidden', mode !== 'single');
    groups.triangle.classList.toggle('hidden', mode !== 'triangle');
    groups.sun.classList.toggle('hidden', !['target', 'train', 'compare'].includes(mode));
    groups.train.classList.toggle('hidden', mode !== 'train');
    groups.compare.classList.toggle('hidden', mode !== 'compare');
  }

  function applyActivation(z, kind) {
    if (kind === 'relu') return Math.max(0, z);
    if (kind === 'sigmoid') return NN.sigmoid(z);
    if (kind === 'tanh') return (Math.tanh(z) + 1) / 2;
    const scaled = 0.5 + Math.atan(z) / Math.PI;
    return scaled;
  }

  function drawBase() {
    Draw.clear(ctx, canvas);
    Draw.drawGrid(ctx, canvas);
  }

  function drawSquareAndRays(showFill = true) {
    for (const tri of rays) {
      Draw.drawTriangle(ctx, canvas, tri, showFill ? 'rgba(255, 190, 74, 0.32)' : 'rgba(255,255,255,0)', '#d18100');
    }
    Draw.drawPolygon(ctx, canvas, Geo.squareVertices(), 'rgba(215,215,230,0.18)', '#77798e');
  }

  function drawSingle() {
    drawBase();
    const w1 = parseFloat(w1Slider.value);
    const w2 = parseFloat(w2Slider.value);
    const b = parseFloat(bSlider.value);
    const activation = singleActivation.value;
    Draw.drawScalarField(ctx, canvas, (x, y) => applyActivation(w1 * x + w2 * y + b, activation), 90);
    Draw.drawGrid(ctx, canvas);
    Draw.drawLineForPlane(ctx, canvas, w1, w2, b, '#1a1a1a');
    statusEl.innerHTML = `Neuron equation: z = ${w1.toFixed(2)}x + ${w2.toFixed(2)}y + ${b.toFixed(2)}. The dark line is z = 0. The color field shows the value after ${activation}.`;
    explanationEl.textContent = 'A single neuron creates a boundary. Changing w1 and w2 rotates it. Changing b shifts it. ReLU turns one side off; sigmoid makes a soft transition.';
  }

  function drawTriangleMode() {
    drawBase();
    const tri = [
      { x: -0.42, y: -0.45 },
      { x: 0.00, y: 0.68 },
      { x: 0.42, y: -0.45 },
    ];
    const planes = Geo.inwardHalfPlanesForTriangle(tri);
    const show = triangleShow.value;
    if (show === 'all') {
      Draw.drawScalarField(ctx, canvas, (x, y) => {
        const active = planes.every(p => Geo.planeValue(p, x, y) >= 0);
        return active ? 1 : 0;
      }, 90);
    } else {
      const index = parseInt(show.replace('edge', ''), 10);
      const plane = planes[index];
      Draw.drawScalarField(ctx, canvas, (x, y) => Geo.planeValue(plane, x, y) >= 0 ? 1 : 0, 90);
    }
    Draw.drawGrid(ctx, canvas);
    Draw.drawTriangle(ctx, canvas, tri, 'rgba(255,190,74,0.22)', '#d18100');
    planes.forEach((p, i) => Draw.drawLineForPlane(ctx, canvas, p.w1, p.w2, p.b, i === 0 ? '#1c2b7a' : i === 1 ? '#7a1c54' : '#1f7a3b'));
    statusEl.innerHTML = 'A triangle is the intersection of three half-planes. Each side can be interpreted as one boundary test.';
    explanationEl.textContent = 'Layer 1 can create the three edge tests. A later layer can combine those tests into an inside-triangle detector.';
  }

  function drawTarget() {
    drawBase();
    Draw.drawScalarField(ctx, canvas, (x, y) => Geo.labelPoint({ x, y }, rays), 95);
    Draw.drawGrid(ctx, canvas);
    drawSquareAndRays(false);
    Draw.drawPoints(ctx, canvas, dataset);
    Draw.drawLegend(ctx, canvas);
    statusEl.innerHTML = `${rays.length} triangular rays. Target label is 1 only inside a ray, 0 outside. The square is just context.`;
    explanationEl.textContent = 'The sun-ray region is a union of triangles. For 8 rays, the conceptual design is 24 boundary-like units, 8 ray detectors, and 1 final union detector.';
  }

  function drawTrain() {
    drawBase();
    Draw.drawScalarField(ctx, canvas, (x, y) => model.predict(x, y), 80);
    Draw.drawGrid(ctx, canvas);
    drawSquareAndRays(false);
    Draw.drawPoints(ctx, canvas, dataset);
    const metrics = lastMetrics || model.lossAndAccuracy(dataset);
    const archText = model.sizes.join(' -> ');
    statusEl.innerHTML = `Architecture: ${archText}. Parameters: ${model.parameterCount()}. Steps: ${model.stepCount}. Loss: ${metrics.loss.toFixed(4)}. Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%.`;
    explanationEl.textContent = 'The orange probability field is what the network currently believes is inside a ray. Training moves and weights boundaries until this field resembles the target triangles.';
  }

  function drawCompare() {
    drawBase();
    drawSquareAndRays(true);
    Draw.drawPoints(ctx, canvas, dataset);
    Draw.drawLegend(ctx, canvas);
    statusEl.innerHTML = 'Click Run comparison to train several architectures on the same dataset.';
    explanationEl.textContent = 'Compare depth and width: width gives more primitive pieces, depth gives more stages of composition.';
  }

  function draw() {
    updateControlVisibility();
    const mode = modeSelect.value;
    lrValue.textContent = parseFloat(learningRate.value).toFixed(3);
    if (mode === 'single') drawSingle();
    else if (mode === 'triangle') drawTriangleMode();
    else if (mode === 'target') drawTarget();
    else if (mode === 'train') drawTrain();
    else if (mode === 'compare') drawCompare();
  }

  function trainSteps(count) {
    const lr = parseFloat(learningRate.value);
    for (let i = 0; i < count; i += 1) {
      lastMetrics = model.trainStep(dataset, lr, 160);
    }
    lastMetrics = model.lossAndAccuracy(dataset);
    draw();
  }

  function resetModel() {
    model = makeModel();
    lastMetrics = null;
    draw();
  }

  function runComparison() {
    const configs = [
      '2,8,1',
      '2,24,8,1',
      '2,64,1',
      '2,16,16,16,1',
    ];
    const rows = [];
    for (const cfg of configs) {
      const m = new NN.MLP(NN.parseArchitecture(cfg), hiddenActivation.value);
      for (let i = 0; i < 900; i += 1) {
        m.trainStep(dataset, parseFloat(learningRate.value), 160);
      }
      const metrics = m.lossAndAccuracy(dataset);
      rows.push({ cfg, params: m.parameterCount(), loss: metrics.loss, acc: metrics.accuracy });
    }
    compareResults.innerHTML = `
      <table>
        <thead><tr><th>Architecture</th><th>Params</th><th>Loss</th><th>Accuracy</th></tr></thead>
        <tbody>
          ${rows.map(r => `<tr><td>${r.cfg.replaceAll(',', ' -> ')}</td><td>${r.params}</td><td>${r.loss.toFixed(3)}</td><td>${(100 * r.acc).toFixed(1)}%</td></tr>`).join('')}
        </tbody>
      </table>
    `;
    statusEl.innerHTML = 'Comparison complete. The numbers can vary because each model starts with random weights.';
  }

  [modeSelect, w1Slider, w2Slider, bSlider, singleActivation, triangleShow, rayCountSelect, sampleCountSelect, architectureSelect, hiddenActivation, learningRate].forEach(el => {
    el.addEventListener('input', () => {
      if ([rayCountSelect, sampleCountSelect].includes(el)) regenerateDataset();
      else if ([architectureSelect, hiddenActivation].includes(el)) resetModel();
      else draw();
    });
  });

  newSamplesBtn.addEventListener('click', regenerateDataset);
  resetModelBtn.addEventListener('click', resetModel);
  trainOneBtn.addEventListener('click', () => trainSteps(1));
  train100Btn.addEventListener('click', () => trainSteps(100));
  train1000Btn.addEventListener('click', () => trainSteps(1000));
  runCompareBtn.addEventListener('click', runComparison);

  draw();
})();
