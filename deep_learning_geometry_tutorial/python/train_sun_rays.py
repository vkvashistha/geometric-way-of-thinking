"""Train a small MLP to classify points inside triangular sun rays."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from geometry import make_sun_rays, sample_points, square_vertices
from mlp_from_scratch import MLP


def plot_decision(model: MLP, x: np.ndarray, y: np.ndarray, rays, output_path: Path, title: str) -> None:
    xs = np.linspace(-1.1, 1.1, 260)
    ys = np.linspace(-1.1, 1.1, 260)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = model.predict_proba(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 7))
    plt.imshow(proba, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", alpha=0.70, vmin=0, vmax=1)
    plt.colorbar(label="model probability: inside ray")

    for tri in rays:
        closed = np.vstack([tri.vertices, tri.vertices[0]])
        plt.plot(closed[:, 0], closed[:, 1], linewidth=1.8)

    square = square_vertices()
    closed_square = np.vstack([square, square[0]])
    plt.plot(closed_square[:, 0], closed_square[:, 1], linewidth=1.5)

    pos = y == 1
    neg = ~pos
    plt.scatter(x[neg, 0], x[neg, 1], s=8, alpha=0.35, label="outside ray")
    plt.scatter(x[pos, 0], x[pos, 1], s=8, alpha=0.65, label="inside ray")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    rays = make_sun_rays(ray_count=8)
    x_train, y_train = sample_points(1600, rays, seed=3)
    x_test, y_test = sample_points(1200, rays, seed=4)

    model = MLP([2, 24, 8, 1], activation="relu", seed=1)

    history = []
    steps = 2500
    for step in range(steps):
        batch_loss, batch_acc = model.train_step(x_train, y_train, learning_rate=0.08, batch_size=160)
        if step % 50 == 0 or step == steps - 1:
            train_loss, train_acc = model.loss_and_accuracy(x_train, y_train)
            test_loss, test_acc = model.loss_and_accuracy(x_test, y_test)
            history.append((step, train_loss, train_acc, test_loss, test_acc))
            print(
                f"step={step:4d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
            )

    plot_decision(
        model,
        x_test,
        y_test.astype(bool),
        rays,
        out_dir / "sun_rays_decision_boundary.png",
        title="MLP decision field for triangular sun rays",
    )

    hist = np.array(history)
    plt.figure(figsize=(8, 5))
    plt.plot(hist[:, 0], hist[:, 1], label="train loss")
    plt.plot(hist[:, 0], hist[:, 3], label="test loss")
    plt.xlabel("training step")
    plt.ylabel("binary cross-entropy")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sun_rays_loss_curve.png", dpi=160)
    plt.close()

    print(f"parameters: {model.parameter_count}")
    print(f"saved {out_dir / 'sun_rays_decision_boundary.png'}")
    print(f"saved {out_dir / 'sun_rays_loss_curve.png'}")


if __name__ == "__main__":
    main()
