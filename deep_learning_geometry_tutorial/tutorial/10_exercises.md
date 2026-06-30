# 10 - Exercises

Use the browser simulation or Python scripts to answer these.

---

## Exercise 1: Single neuron

In the browser simulation, choose `Single neuron boundary`.

Move the sliders for:

```text
w1
w2
b
```

Questions:

1. Which parameters rotate the boundary?
2. Which parameter shifts the boundary?
3. What changes when you switch from no activation to ReLU?
4. What changes when you switch from ReLU to sigmoid?

---

## Exercise 2: Three boundaries make a triangle

Choose `Three boundaries make a triangle`.

Questions:

1. How many half-plane tests are required for one triangle?
2. Why is a triangle an intersection rather than a union?
3. What happens when one edge test is removed?

---

## Exercise 3: Sun rays

Choose `Sun-ray target`.

Try:

```text
4 rays
8 rays
12 rays
```

Questions:

1. How many boundary-like pieces would you expect for each ray count?
2. Why does 12 rays need more capacity than 4 rays?

---

## Exercise 4: Training

Choose `Trainable neural network`.

Train these architectures:

```text
2,8,1
2,24,8,1
2,64,1
2,16,16,16,1
```

Questions:

1. Which learns fastest?
2. Which gives the cleanest decision boundary?
3. Which seems to overfit or create unnecessary regions?

---

## Exercise 5: Activation comparison

Train the same architecture with:

```text
ReLU
sigmoid
tanh
```

Questions:

1. Which gives the sharpest-looking boundary?
2. Which gives the smoothest probability field?
3. Which trains most reliably for this problem?

---

## Exercise 6: Build your own shape

Modify `python/geometry.py` or `html_sim/js/geometry.js`.

Try creating:

```text
only one triangle
four corner triangles
an X shape
multiple disjoint squares
```

Then ask:

```text
How many primitive boundaries does the shape need?
How many combination stages does it need?
```
