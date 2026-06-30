"""A tiny neural network implementation using only NumPy.

This is intentionally educational rather than optimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def activate(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(0, z)
    if kind == "sigmoid":
        return sigmoid(z)
    if kind == "tanh":
        return np.tanh(z)
    if kind == "linear":
        return z
    raise ValueError(f"unknown activation: {kind}")


def activation_derivative(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return (z > 0).astype(float)
    if kind == "sigmoid":
        s = sigmoid(z)
        return s * (1 - s)
    if kind == "tanh":
        t = np.tanh(z)
        return 1 - t * t
    if kind == "linear":
        return np.ones_like(z)
    raise ValueError(f"unknown activation: {kind}")


@dataclass
class ForwardCache:
    activations: list[np.ndarray]
    zs: list[np.ndarray]


class MLP:
    """Small fully connected binary classifier."""

    def __init__(self, sizes: Sequence[int], activation: str = "relu", seed: int | None = None):
        if len(sizes) < 2:
            raise ValueError("sizes must contain input and output sizes")
        if sizes[0] != 2 or sizes[-1] != 1:
            raise ValueError("this tutorial expects sizes like [2, 24, 8, 1]")

        self.sizes = list(sizes)
        self.activation = activation
        self.rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.steps = 0
        self.reset()

    def reset(self) -> None:
        self.weights = []
        self.biases = []
        self.steps = 0
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(self.rng.normal(0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
        a = x
        activations = [a]
        zs: list[np.ndarray] = []

        for layer_index, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            zs.append(z)
            is_output = layer_index == len(self.weights) - 1
            a = sigmoid(z) if is_output else activate(z, self.activation)
            activations.append(a)

        return a, ForwardCache(activations=activations, zs=zs)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        y_hat, _ = self.forward(x)
        return y_hat[:, 0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(float)

    def loss_and_accuracy(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        y = y.reshape(-1, 1)
        y_hat, _ = self.forward(x)
        eps = 1e-8
        loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        accuracy = np.mean((y_hat[:, 0] >= 0.5) == (y[:, 0] >= 0.5))
        return float(loss), float(accuracy)

    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.05, batch_size: int = 128) -> tuple[float, float]:
        n = x.shape[0]
        idx = self.rng.choice(n, size=min(batch_size, n), replace=False)
        xb = x[idx]
        yb = y[idx].reshape(-1, 1)
        m = xb.shape[0]

        y_hat, cache = self.forward(xb)
        eps = 1e-8
        loss = -np.mean(yb * np.log(y_hat + eps) + (1 - yb) * np.log(1 - y_hat + eps))
        accuracy = np.mean((y_hat[:, 0] >= 0.5) == (yb[:, 0] >= 0.5))

        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        # Sigmoid output + binary cross-entropy derivative.
        delta = y_hat - yb

        for layer_index in reversed(range(len(self.weights))):
            a_prev = cache.activations[layer_index]
            grad_w[layer_index] = a_prev.T @ delta / m
            grad_b[layer_index] = delta.mean(axis=0, keepdims=True)

            if layer_index > 0:
                delta = (delta @ self.weights[layer_index].T) * activation_derivative(
                    cache.zs[layer_index - 1], self.activation
                )

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i] -= learning_rate * grad_b[i]

        self.steps += 1
        return float(loss), float(accuracy)

    @property
    def parameter_count(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))
