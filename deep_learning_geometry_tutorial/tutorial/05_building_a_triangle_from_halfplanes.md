# 05 - Building a Triangle from Half-Planes

A triangle is a perfect example because it is made from three straight sides.

```text
       /\
      /  \
     /____\
```

Each side defines a line.

Each line divides space into two half-planes.

The triangle is the intersection of three correct half-planes.

---

## One triangle as logic

For a point `(x, y)`, ask three questions:

```text
Q1: Is the point inside the left edge?
Q2: Is the point inside the right edge?
Q3: Is the point above the base edge?
```

Then:

```text
inside triangle = Q1 AND Q2 AND Q3
```

---

## Neural-network version

Layer 1:

```text
Neuron 1 -> boundary for left edge
Neuron 2 -> boundary for right edge
Neuron 3 -> boundary for base edge
```

Layer 2:

```text
Triangle detector -> combines the three edge activations
```

Architecture:

```text
Input: (x, y)
   |
   v
3 boundary neurons
   |
   v
1 triangle-region neuron
```

---

## Important caveat

A real trained ReLU network does not necessarily discover exactly these three textbook lines.

It may use extra pieces, tilted pieces, or approximate logic.

But conceptually, this is the cleanest way to understand why a triangle is natural for a neural network:

```text
triangle = combination of three boundary tests
```
