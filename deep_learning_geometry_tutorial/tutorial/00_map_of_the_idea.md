# 00 - Map of the Idea

This tutorial explains deep learning through geometry.

Instead of starting with tensors, chain rule, or backpropagation, we start with this question:

> How can many simple straight boundaries combine into a complex shape?

Our final example is a simple sun drawing:

```text
       /\
   /\ /  \ /\
  /  \____/  \
  |          |
  |  square  |
  |          |
  \  /----\  /
   \/      \/
```

The model receives a point:

```text
(x, y)
```

and predicts:

```text
1 = inside a triangular ray
0 = outside the triangular rays
```

For simplicity, the center can be treated as a square, while the target region is only the triangular rays around it.

---

## The full mental model

```text
Linear model
  -> one flat line, plane, or hyperplane

Neuron
  -> creates one learned boundary

Activation function
  -> gates or softens the boundary

Many neurons
  -> many boundaries and many regions

Hidden layer
  -> a collection of boundary/feature detectors

Next hidden layer
  -> combines previous boundaries/features into larger regions

Training
  -> moves, rotates, and weights boundaries until the output matches data
```

---

## The key slogan

```text
Deep learning = geometry + composition + optimization
```

Geometry gives us boundaries.

Composition lets layers build complex shapes from simple shapes.

Optimization adjusts the parameters so the shape fits the data.

---

## The important correction

It is tempting to say:

```text
More hidden layers make the curve smoother.
```

That is not exactly right.

A better statement is:

```text
More neurons give more primitive pieces.
More layers give more stages of combination.
```

With ReLU, the learned function is still piecewise-linear. It can look smooth when there are many tiny pieces, just like a many-sided polygon can look like a circle.
