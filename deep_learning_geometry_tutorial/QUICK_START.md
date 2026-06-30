# Quick Start

## 1. Read the tutorial

Open the files in this order:

```text
tutorial/00_map_of_the_idea.md
tutorial/01_curve_fitting_to_hyperplanes.md
tutorial/02_neuron_as_boundary.md
tutorial/03_relu_as_gate.md
tutorial/04_many_neurons_many_regions.md
tutorial/05_building_a_triangle_from_halfplanes.md
tutorial/06_sun_rays_union_of_triangles.md
tutorial/07_smoothness_depth_width.md
tutorial/08_training_as_moving_boundaries.md
tutorial/09_how_to_choose_layers_and_neurons.md
tutorial/10_exercises.md
```

## 2. Run the browser simulation

Open this file in a browser:

```text
html_sim/index.html
```

No server is required. It is a static HTML/JS simulation.

Use these modes:

```text
Single neuron boundary
Three boundaries make a triangle
Sun-ray target
Trainable neural network
Depth vs width comparison
```

## 3. Run the Python version

Create an environment and install requirements:

```bash
pip install -r requirements.txt
```

Run:

```bash
cd python
python plot_single_neuron.py
python train_sun_rays.py
python compare_architectures.py
```

Generated plots are saved in:

```text
python/outputs/
```

## 4. Main idea to remember

```text
A neuron does not draw the final shape by itself.
It creates a simple boundary.
Layers combine these simple boundaries into regions.
Training moves and weights those boundaries until the target region appears.
```
