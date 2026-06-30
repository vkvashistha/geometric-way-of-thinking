"""Visualize a single neuron boundary and activation field."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mlp_from_scratch import sigmoid, activate


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    w1, w2, b = 1.2, -0.8, 0.0
    activation = "relu"

    xs = np.linspace(-1.1, 1.1, 250)
    ys = np.linspace(-1.1, 1.1, 250)
    xx, yy = np.meshgrid(xs, ys)
    z = w1 * xx + w2 * yy + b

    if activation == "linear":
        field = z
    elif activation == "sigmoid":
        field = sigmoid(z)
    elif activation == "tanh":
        field = np.tanh(z)
    else:
        field = activate(z, activation)

    plt.figure(figsize=(7, 7))
    plt.imshow(field, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", alpha=0.75)
    plt.colorbar(label=f"activation: {activation}")
    plt.contour(xx, yy, z, levels=[0], linewidths=2)
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.title(f"Single neuron: z = {w1}x + {w2}y + {b}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_dir / "single_neuron_boundary.png", dpi=160)
    print(f"saved {out_dir / 'single_neuron_boundary.png'}")


if __name__ == "__main__":
    main()
