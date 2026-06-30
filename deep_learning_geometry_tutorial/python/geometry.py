"""Geometry helpers for the sun-ray classification task.

The target label is 1 when a point is inside any triangular ray and 0 otherwise.
The center square is drawn for visual context, but it is not part of the positive
class unless you modify label_points().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math
import numpy as np


@dataclass(frozen=True)
class Triangle:
    """Triangle represented by three vertices with shape (3, 2)."""

    vertices: np.ndarray


def make_sun_rays(ray_count: int = 8, inner_radius: float = 0.48, outer_radius: float = 0.88) -> list[Triangle]:
    """Create triangular rays around the origin.

    Parameters
    ----------
    ray_count:
        Number of triangular rays.
    inner_radius:
        Radius of the triangle bases.
    outer_radius:
        Radius of the triangle tips.
    """

    if ray_count < 1:
        raise ValueError("ray_count must be positive")

    base_width = min(0.38, 2 * inner_radius * math.sin(math.pi / ray_count) * 0.95)
    rays: list[Triangle] = []

    for i in range(ray_count):
        theta = (2 * math.pi * i) / ray_count - math.pi / 2
        direction = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        tangent = np.array([-direction[1], direction[0]], dtype=float)
        base_center = inner_radius * direction
        apex = outer_radius * direction
        half = base_width / 2
        p1 = base_center + half * tangent
        p2 = apex
        p3 = base_center - half * tangent
        rays.append(Triangle(np.vstack([p1, p2, p3])))

    return rays


def triangle_area_sign(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed area-like value used by the point-in-triangle test."""

    return (points[:, 0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (points[:, 1] - b[1])


def points_in_triangle(points: np.ndarray, triangle: Triangle) -> np.ndarray:
    """Return a boolean mask for points inside a triangle."""

    vertices = triangle.vertices
    a, b, c = vertices[0], vertices[1], vertices[2]
    d1 = triangle_area_sign(points, a, b)
    d2 = triangle_area_sign(points, b, c)
    d3 = triangle_area_sign(points, c, a)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def label_points(points: np.ndarray, rays: Iterable[Triangle]) -> np.ndarray:
    """Label points as 1 if inside any triangular ray, otherwise 0."""

    points = np.asarray(points, dtype=float)
    labels = np.zeros(points.shape[0], dtype=float)
    for triangle in rays:
        labels[points_in_triangle(points, triangle)] = 1.0
    return labels


def sample_points(
    n: int,
    rays: Iterable[Triangle],
    low: float = -1.05,
    high: float = 1.05,
    seed: int | None = None,
    balanced: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random 2D points and label them.

    By default this uses balanced sampling: about half of the points are
    inside a ray and half are outside. This avoids the misleading baseline
    where a model predicts "outside" everywhere and still gets high accuracy
    because the positive ray region is small.
    """

    ray_list = list(rays)
    rng = np.random.default_rng(seed)

    if not balanced:
        points = rng.uniform(low, high, size=(n, 2))
        labels = label_points(points, ray_list)
        return points, labels

    target_pos = n // 2
    target_neg = n - target_pos
    pos_points: list[np.ndarray] = []
    neg_points: list[np.ndarray] = []
    attempts = 0
    max_attempts = n * 300

    while (len(pos_points) < target_pos or len(neg_points) < target_neg) and attempts < max_attempts:
        attempts += 1
        point = rng.uniform(low, high, size=(1, 2))
        label = label_points(point, ray_list)[0]
        if label == 1 and len(pos_points) < target_pos:
            pos_points.append(point[0])
        elif label == 0 and len(neg_points) < target_neg:
            neg_points.append(point[0])

    if len(pos_points) < target_pos or len(neg_points) < target_neg:
        raise RuntimeError("could not generate enough balanced samples; try fewer rays or more attempts")

    points = np.vstack([pos_points, neg_points])
    labels = np.r_[np.ones(len(pos_points)), np.zeros(len(neg_points))]
    order = rng.permutation(points.shape[0])
    return points[order], labels[order]


def square_vertices(half_side: float = 0.34) -> np.ndarray:
    """Return vertices for the visual center square."""

    return np.array(
        [
            [-half_side, -half_side],
            [half_side, -half_side],
            [half_side, half_side],
            [-half_side, half_side],
        ],
        dtype=float,
    )


def inward_halfplanes_for_triangle(triangle: Triangle) -> list[tuple[float, float, float]]:
    """Return oriented half-planes w1*x + w2*y + b >= 0 for a triangle.

    These are useful for showing how one triangle can be viewed as three
    boundary tests.
    """

    vertices = triangle.vertices
    centroid = vertices.mean(axis=0)
    planes: list[tuple[float, float, float]] = []

    for i in range(3):
        a = vertices[i]
        b_point = vertices[(i + 1) % 3]
        w1 = b_point[1] - a[1]
        w2 = -(b_point[0] - a[0])
        bias = -(w1 * a[0] + w2 * a[1])
        inside_value = w1 * centroid[0] + w2 * centroid[1] + bias
        if inside_value < 0:
            w1, w2, bias = -w1, -w2, -bias
        planes.append((float(w1), float(w2), float(bias)))

    return planes
