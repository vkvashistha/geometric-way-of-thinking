# 06 - Sun Rays as a Union of Triangles

Now combine many triangles.

Suppose the sun has 8 triangular rays.

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

The target region is:

```text
inside any triangular ray
```

not necessarily inside the square center.

---

## Geometry

One triangle:

```text
3 boundary tests
```

Eight triangles:

```text
8 x 3 = 24 boundary tests
```

Then combine each group of three boundaries into one triangle detector:

```text
8 triangle detectors
```

Then combine the triangle detectors into the final output:

```text
inside ray 1 OR inside ray 2 OR ... OR inside ray 8
```

---

## Conceptual architecture

```text
Input: (x, y)
   |
   v
Layer 1: 24 boundary-like neurons
   |
   v
Layer 2: 8 triangle-like neurons
   |
   v
Output: 1 sun-ray detector
```

Written as layer sizes:

```text
2 -> 24 -> 8 -> 1
```

---

## Why this is useful

This architecture mirrors the structure of the problem.

```text
low-level pieces: edges
mid-level pieces: triangles
high-level region: union of rays
```

That is the larger lesson of deep learning:

```text
Good architectures often reflect the structure of the data.
```

For images, CNNs reflect spatial locality.

For text, Transformers reflect token-to-token relationships.

For this toy geometry problem, a boundary-to-triangle-to-union architecture reflects the shape itself.
