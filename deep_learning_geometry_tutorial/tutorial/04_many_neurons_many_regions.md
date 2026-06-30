# 04 - Many Neurons Create Many Regions

One neuron creates one boundary.

Many neurons create many boundaries.

```text
\   |   /
 \  |  /
--\ | /-----
   \|/
----/|------
  / | \
 /  |  \
```

These boundaries partition the input space into regions.

---

## Width

The number of neurons in a layer is called the width of that layer.

A wider layer can create more primitive pieces.

```text
2 neurons   -> few boundaries
24 neurons  -> many boundaries
128 neurons -> many more boundaries
```

This does not automatically mean better generalization. It means more capacity.

---

## What the next layer sees

The next layer does not directly see only raw `(x, y)`.

It sees activation values:

```text
h1 = how strongly boundary 1 fired
h2 = how strongly boundary 2 fired
h3 = how strongly boundary 3 fired
...
```

So the network gradually replaces the original coordinate system with learned coordinates.

```text
Original coordinates:
  (x, y)

Learned coordinates:
  (activation of neuron 1,
   activation of neuron 2,
   activation of neuron 3,
   ...)
```

---

## Important geometric idea

A hidden layer can be understood as a learned feature map:

```text
(x, y) -> [h1, h2, h3, ..., hk]
```

Each `hi` is a learned question about the input.

The next layer combines the answers to those questions.
