# 03 - ReLU as a Gate

The ReLU activation is:

```math
h = max(0, z)
```

It behaves like this:

```text
z < 0  ->  h = 0
z > 0  ->  h = z
```

So ReLU acts like a gate.

---

## Geometric meaning

First the neuron creates a boundary:

```math
z = w_1x + w_2y + b
```

Then ReLU says:

```text
One side contributes nothing.
The other side passes forward.
```

Diagram:

```text
inactive side       active side
output = 0          output = z

      - - - - | + + + +
      - - - - | + + + +
      - - - - | + + + +
```

---

## Why this matters for hidden layers

Suppose the first hidden layer has five neurons:

```text
h = [2.3, 0.0, 1.7, 0.0, 5.1]
```

Then the next layer receives:

```text
neuron 1 active
neuron 2 off
neuron 3 active
neuron 4 off
neuron 5 active
```

This matches the intuition:

```text
Some boundaries are available.
Some boundaries are not.
The next layer weights the active ones.
```

---

## ReLU is sharp, sigmoid is soft

ReLU is like a hard one-sided gate, although the positive side remains linear.

Sigmoid is a softer gate:

```math
sigma(z) = 1 / (1 + e^{-z})
```

It gradually moves from 0 to 1.

```text
ReLU:    sharp hinge
Sigmoid: smooth transition
Tanh:    smooth transition from -1 to 1
```

The simulation lets you compare these activations visually.
