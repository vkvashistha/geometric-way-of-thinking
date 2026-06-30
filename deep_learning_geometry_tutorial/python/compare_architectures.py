"""Compare shallow, wide, and deep architectures on the sun-ray task."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from geometry import make_sun_rays, sample_points, square_vertices
from mlp_from_scratch import MLP


def plot_model(model: MLP, rays, output_path: Path, title: str) -> None:
    xs = np.linspace(-1.1, 1.1, 220)
    ys = np.linspace(-1.1, 1.1, 220)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = model.predict_proba(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.imshow(proba, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", alpha=0.75, vmin=0, vmax=1)
    for tri in rays:
        closed = np.vstack([tri.vertices, tri.vertices[0]])
        plt.plot(closed[:, 0], closed[:, 1], linewidth=1.2)
    square = square_vertices()
    closed_square = np.vstack([square, square[0]])
    plt.plot(closed_square[:, 0], closed_square[:, 1], linewidth=1.2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    rays = make_sun_rays(ray_count=8)
    x_train, y_train = sample_points(1800, rays, seed=10)
    x_test, y_test = sample_points(1400, rays, seed=11)

    configs = [
        [2, 8, 1],
        [2, 24, 8, 1],
        [2, 64, 1],
        [2, 16, 16, 16, 1],
    ]

    results = []
    for idx, config in enumerate(configs):
        model = MLP(config, activation="relu", seed=100 + idx)
        for _ in range(2200):
            model.train_step(x_train, y_train, learning_rate=0.08, batch_size=160)

        train_loss, train_acc = model.loss_and_accuracy(x_train, y_train)
        test_loss, test_acc = model.loss_and_accuracy(x_test, y_test)
        results.append((config, model.parameter_count, train_loss, train_acc, test_loss, test_acc))

        filename = "decision_" + "_".join(str(v) for v in config) + ".png"
        plot_model(model, rays, out_dir / filename, title=" -> ".join(str(v) for v in config))

    print("architecture, parameters, train_loss, train_acc, test_loss, test_acc")
    for config, params, train_loss, train_acc, test_loss, test_acc in results:
        arch = " -> ".join(str(v) for v in config)
        print(f"{arch}, {params}, {train_loss:.4f}, {train_acc:.3f}, {test_loss:.4f}, {test_acc:.3f}")

    labels = [" -> ".join(str(v) for v in r[0]) for r in results]
    accuracies = [r[5] for r in results]

    plt.figure(figsize=(9, 5))
    positions = np.arange(len(labels))
    plt.bar(positions, accuracies)
    plt.xticks(positions, labels, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("test accuracy")
    plt.title("Architecture comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "architecture_comparison.png", dpi=160)
    plt.close()

    print(f"saved plots in {out_dir}")


if __name__ == "__main__":
    main()
