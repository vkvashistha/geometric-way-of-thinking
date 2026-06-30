/* A small neural network from scratch for browser visualization.
   The network supports binary classification with sigmoid output and
   ReLU/sigmoid/tanh hidden activations. */

const NN = (() => {
  function randn() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function sigmoid(x) {
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
  }

  function activate(x, kind) {
    if (kind === 'relu') return Math.max(0, x);
    if (kind === 'sigmoid') return sigmoid(x);
    if (kind === 'tanh') return Math.tanh(x);
    return x;
  }

  function activationDerivativeFromZ(z, kind) {
    if (kind === 'relu') return z > 0 ? 1 : 0;
    if (kind === 'sigmoid') {
      const s = sigmoid(z);
      return s * (1 - s);
    }
    if (kind === 'tanh') {
      const t = Math.tanh(z);
      return 1 - t * t;
    }
    return 1;
  }

  class MLP {
    constructor(sizes, activation = 'relu') {
      this.sizes = sizes.slice();
      this.activation = activation;
      this.stepCount = 0;
      this.weights = [];
      this.biases = [];
      this.reset();
    }

    reset() {
      this.weights = [];
      this.biases = [];
      this.stepCount = 0;
      for (let layer = 1; layer < this.sizes.length; layer += 1) {
        const fanIn = this.sizes[layer - 1];
        const fanOut = this.sizes[layer];
        const scale = Math.sqrt(2 / fanIn);
        const w = [];
        const b = [];
        for (let j = 0; j < fanOut; j += 1) {
          const row = [];
          for (let i = 0; i < fanIn; i += 1) {
            row.push(randn() * scale);
          }
          w.push(row);
          b.push(0);
        }
        this.weights.push(w);
        this.biases.push(b);
      }
    }

    forwardVector(input) {
      let a = input.slice();
      const activations = [a];
      const zs = [];
      for (let layer = 0; layer < this.weights.length; layer += 1) {
        const w = this.weights[layer];
        const b = this.biases[layer];
        const z = [];
        const nextA = [];
        const isOutput = layer === this.weights.length - 1;
        for (let j = 0; j < w.length; j += 1) {
          let value = b[j];
          for (let i = 0; i < w[j].length; i += 1) {
            value += w[j][i] * a[i];
          }
          z.push(value);
          nextA.push(isOutput ? sigmoid(value) : activate(value, this.activation));
        }
        zs.push(z);
        activations.push(nextA);
        a = nextA;
      }
      return { output: a[0], activations, zs };
    }

    predict(x, y) {
      return this.forwardVector([x, y]).output;
    }

    lossAndAccuracy(dataset) {
      let loss = 0;
      let correct = 0;
      const eps = 1e-8;
      for (const p of dataset) {
        const yHat = this.predict(p.x, p.y);
        loss += -(p.label * Math.log(yHat + eps) + (1 - p.label) * Math.log(1 - yHat + eps));
        const pred = yHat >= 0.5 ? 1 : 0;
        if (pred === p.label) correct += 1;
      }
      return { loss: loss / dataset.length, accuracy: correct / dataset.length };
    }

    trainStep(dataset, learningRate = 0.05, batchSize = 128) {
      const gradW = this.weights.map(w => w.map(row => row.map(() => 0)));
      const gradB = this.biases.map(b => b.map(() => 0));
      const n = Math.min(batchSize, dataset.length);
      let batchLoss = 0;
      let correct = 0;
      const eps = 1e-8;

      for (let sampleIndex = 0; sampleIndex < n; sampleIndex += 1) {
        const p = dataset[Math.floor(Math.random() * dataset.length)];
        const y = p.label;
        const cache = this.forwardVector([p.x, p.y]);
        const yHat = cache.output;
        batchLoss += -(y * Math.log(yHat + eps) + (1 - y) * Math.log(1 - yHat + eps));
        if ((yHat >= 0.5 ? 1 : 0) === y) correct += 1;

        const deltas = new Array(this.weights.length);
        deltas[deltas.length - 1] = [yHat - y];

        for (let layer = this.weights.length - 2; layer >= 0; layer -= 1) {
          const nextW = this.weights[layer + 1];
          const nextDelta = deltas[layer + 1];
          const z = cache.zs[layer];
          const delta = [];
          for (let i = 0; i < this.sizes[layer + 1]; i += 1) {
            let sum = 0;
            for (let j = 0; j < nextDelta.length; j += 1) {
              sum += nextW[j][i] * nextDelta[j];
            }
            delta.push(sum * activationDerivativeFromZ(z[i], this.activation));
          }
          deltas[layer] = delta;
        }

        for (let layer = 0; layer < this.weights.length; layer += 1) {
          const aPrev = cache.activations[layer];
          for (let j = 0; j < this.weights[layer].length; j += 1) {
            gradB[layer][j] += deltas[layer][j];
            for (let i = 0; i < this.weights[layer][j].length; i += 1) {
              gradW[layer][j][i] += deltas[layer][j] * aPrev[i];
            }
          }
        }
      }

      for (let layer = 0; layer < this.weights.length; layer += 1) {
        for (let j = 0; j < this.weights[layer].length; j += 1) {
          this.biases[layer][j] -= learningRate * gradB[layer][j] / n;
          for (let i = 0; i < this.weights[layer][j].length; i += 1) {
            this.weights[layer][j][i] -= learningRate * gradW[layer][j][i] / n;
          }
        }
      }

      this.stepCount += 1;
      return { loss: batchLoss / n, accuracy: correct / n };
    }

    parameterCount() {
      let count = 0;
      for (let layer = 0; layer < this.weights.length; layer += 1) {
        count += this.biases[layer].length;
        for (const row of this.weights[layer]) count += row.length;
      }
      return count;
    }
  }

  function parseArchitecture(text) {
    const sizes = text.split(',').map(x => parseInt(x.trim(), 10)).filter(Number.isFinite);
    if (sizes.length < 2 || sizes[0] !== 2 || sizes[sizes.length - 1] !== 1) {
      throw new Error('Architecture must look like 2,24,8,1');
    }
    return sizes;
  }

  return { MLP, parseArchitecture, sigmoid, activate };
})();
