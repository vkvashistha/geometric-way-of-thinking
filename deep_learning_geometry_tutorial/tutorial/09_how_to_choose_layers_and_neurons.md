# 09 - How to Choose Layers and Neurons

There is no universal formula for the perfect number of layers and neurons.

But there are useful principles.

---

## Principle 1: Match the structure of the problem

For the sun-ray task:

```text
triangles are made of edges
sun rays are made of triangles
```

So a natural conceptual architecture is:

```text
2 -> 24 -> 8 -> 1
```

for 8 rays.

Why?

```text
2 inputs: x, y
24 boundary-like units: 3 per triangle x 8 triangles
8 triangle-like units: one per ray
1 output: inside any ray
```

---

## Principle 2: Width gives capacity at one level

More neurons in a layer means more learned features at that level.

```text
small width -> may underfit
huge width  -> may overfit or waste computation
```

---

## Principle 3: Depth gives hierarchy

More layers allow the model to reuse and recombine features.

```text
Layer 1: primitive boundaries
Layer 2: small regions
Layer 3: larger patterns
Output: final decision
```

---

## Principle 4: Use validation, not only training accuracy

A model can memorize training points.

So always check performance on new points.

```text
training accuracy high + validation accuracy low -> overfitting
both low -> underfitting or training problem
both high -> good fit
```

---

## Practical starting points for this toy problem

Try:

```text
2 -> 8 -> 1
2 -> 24 -> 8 -> 1
2 -> 64 -> 1
2 -> 16 -> 16 -> 16 -> 1
```

Then compare:

```text
loss
accuracy
decision boundary
number of parameters
```

The browser simulation and Python scripts include this comparison.
