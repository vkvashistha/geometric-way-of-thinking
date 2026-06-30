# 07 - Smoothness, Depth, and Width

A common question is:

> If I add more hidden layers, do I get a smoother curve?

Not necessarily.

---

## ReLU networks are piecewise-linear

With ReLU:

```math
h = max(0, z)
```

the network creates a piecewise-linear function.

That means the learned boundary may look smooth, but it is made from many small linear pieces.

```text
few pieces:

   /\
__/  \__

many pieces:

    __
 __/  \__
/        \__
```

A many-sided polygon can look like a circle, even though it is still made of straight edges.

---

## Smooth activations

If the activation is sigmoid, tanh, or GELU-like, then the activation itself is smooth.

The resulting function can be smoother in a mathematical sense.

But this does not mean smooth activations are always better. ReLU is popular because it is simple, fast, and often trains well.

---

## Width versus depth

A useful mental model:

```text
Width = number of building blocks at one stage
Depth = number of stages of construction
```

A shallow wide model:

```text
2 -> 128 -> 1
```

has many primitive pieces but only one main combination step.

A deeper model:

```text
2 -> 24 -> 8 -> 1
```

has fewer pieces per stage but can combine pieces hierarchically.

---

## Key takeaway

Do not think:

```text
more depth = more smoothness
```

Think:

```text
more width = more primitive pieces
more depth = more stages of composition
```
