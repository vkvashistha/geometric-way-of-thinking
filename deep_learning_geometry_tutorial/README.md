# Deep Learning from Geometry

A self-contained tutorial and simulation package that explains deep learning using the exact path from our conversation:

1. curve fitting
2. linear models as hyperplanes
3. neurons as boundaries
4. ReLU as a gate
5. hidden layers as combinations of boundaries
6. triangles as intersections of half-planes
7. sun rays as unions of triangles
8. training as moving, rotating, and weighting boundaries

The central task is simple:

> Given a 2D point `(x, y)`, predict whether it lies inside one of the triangular sun rays.

This keeps the math visible. The learner can see every boundary, every region, and every learned decision surface.

---

## What is included

```text
deep_learning_geometry_tutorial/
  README.md
  QUICK_START.md
  requirements.txt

  tutorial/
    00_map_of_the_idea.md
    01_curve_fitting_to_hyperplanes.md
    02_neuron_as_boundary.md
    03_relu_as_gate.md
    04_many_neurons_many_regions.md
    05_building_a_triangle_from_halfplanes.md
    06_sun_rays_union_of_triangles.md
    07_smoothness_depth_width.md
    08_training_as_moving_boundaries.md
    09_how_to_choose_layers_and_neurons.md
    10_exercises.md

  html_sim/
    index.html
    css/styles.css
    js/geometry.js
    js/nn.js
    js/draw.js
    js/app.js

  python/
    README.md
    geometry.py
    mlp_from_scratch.py
    plot_single_neuron.py
    train_sun_rays.py
    compare_architectures.py
```

---

## Suggested learning order

Start with the markdown files in `tutorial/`, then open the browser simulation:

```text
html_sim/index.html
```

Then run the Python version:

```bash
cd python
python train_sun_rays.py
python compare_architectures.py
```

---

## What the learner should understand by the end

A neural network can be viewed geometrically:

```text
Neuron        -> creates a boundary
Activation    -> gates one side or softens the boundary
Layer width   -> number of primitive boundaries/features
Layer depth   -> number of stages for combining primitives
Training      -> moves, rotates, and weights boundaries to reduce error
```

For the sun-ray task:

```text
Layer 1: boundary detectors
Layer 2: triangle/ray detectors
Output: union of triangle regions
```

A good conceptual architecture for 8 triangular rays is:

```text
2 -> 24 -> 8 -> 1
```

because:

```text
2 inputs: x, y
24 first-layer neurons: 3 boundary-like pieces per triangle x 8 rays
8 second-layer neurons: one detector per triangular ray
1 output: inside any ray?
```

This is a conceptual architecture, not a strict rule. The training scripts include alternatives so learners can compare width and depth.
