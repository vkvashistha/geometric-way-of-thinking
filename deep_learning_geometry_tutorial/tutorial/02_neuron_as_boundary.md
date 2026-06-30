# 02 - A Neuron as a Boundary

A neuron receives inputs and computes:

```math
z = w_1x + w_2y + b
```

For 2D input, the equation:

```math
w_1x + w_2y + b = 0
```

is a line.

For 3D input, it is a plane.

For higher-dimensional input, it is a hyperplane.

---

## What does the neuron measure?

The neuron does not merely create a line. It tells us which side of the line a point is on.

```text
             z > 0 side
                 +
                 +
-----------------------------  z = 0 boundary
                 -
                 -
             z < 0 side
```

The value of `z` is also related to how far the point is from the boundary, scaled by the weights.

---

## What do weights and bias do?

In:

```math
z = w_1x + w_2y + b
```

`w1` and `w2` control the orientation of the boundary.

`b` shifts the boundary.

So training can be visualized as:

```text
change weights -> rotate boundary
change bias    -> shift boundary
```

---

## Important interpretation

A good sentence is:

```text
A neuron defines a boundary and reports where the input lies relative to that boundary.
```

A slightly less precise sentence is:

```text
A neuron is a hyperplane.
```

The hyperplane is the neuron's zero-boundary. The neuron's output is a measurement relative to that boundary.
