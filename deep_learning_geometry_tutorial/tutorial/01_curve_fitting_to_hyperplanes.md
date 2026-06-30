# 01 - From Curve Fitting to Hyperplanes

Start with the most familiar model:

```math
y = wx + b
```

This creates a straight line.

If the target is curved:

```text
        *
     *     *
   *         *
 *             *
```

a linear model can only try this:

```text
----------------
```

It cannot bend.

---

## Multiple inputs

For two inputs:

```math
y = w_1x_1 + w_2x_2 + b
```

The output is a plane.

For many inputs:

```math
y = W^T X + b
```

The output is a hyperplane.

---

## What is missing?

A flat hyperplane cannot fit complex shapes.

So the question becomes:

```text
How can we build a non-flat shape from flat pieces?
```

A neural network answers:

```text
Create many simple learned pieces.
Gate them.
Weight them.
Combine them in stages.
```

---

## Why this connects to splines

A spline approximates a smooth curve using many smaller pieces.

A ReLU neural network does something similar, but the pieces are learned from data.

```text
Spline:
  manually designed pieces stitched together

Neural network:
  learned pieces stitched together by optimization
```

This is why your Bezier/B-spline intuition is useful. The difference is that neural networks learn their own control pieces instead of requiring you to place them manually.
