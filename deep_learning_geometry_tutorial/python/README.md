# Python Simulations

These scripts mirror the browser simulation.

## Install

From the project root:

```bash
pip install -r requirements.txt
```

## Run

```bash
cd python
python plot_single_neuron.py
python train_sun_rays.py
python compare_architectures.py
```

Outputs are saved into:

```text
python/outputs/
```

## Main files

```text
geometry.py
  Creates triangular sun rays and labels points.

mlp_from_scratch.py
  A tiny neural network implementation using NumPy.

plot_single_neuron.py
  Visualizes one neuron boundary and activation field.

train_sun_rays.py
  Trains an MLP to classify points inside triangular rays.

compare_architectures.py
  Compares shallow/wide/deep models.
```
