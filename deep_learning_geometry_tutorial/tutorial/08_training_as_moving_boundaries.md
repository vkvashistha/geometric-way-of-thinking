# 08 - Training as Moving Boundaries

Before training, the network parameters are random.

So the boundaries are random:

```text
random lines
random gates
random regions
random predictions
```

Training adjusts the parameters to reduce mistakes.

---

## What changes during training?

For a neuron:

```math
z = w_1x + w_2y + b
```

changing `w1` and `w2` rotates the boundary.

changing `b` shifts the boundary.

changing the next-layer weights changes how much this neuron matters.

```text
small output weight -> weak contribution
large output weight -> strong contribution
zero output weight  -> mostly ignored
```

---

## Loss

For classification, a common loss is binary cross-entropy.

Conceptually:

```text
correct confident prediction -> low loss
wrong confident prediction   -> high loss
uncertain prediction         -> medium loss
```

---

## Gradient descent

Gradient descent repeatedly asks:

```text
Which tiny parameter change would reduce the loss?
```

Then it updates:

```text
weights = weights - learning_rate * gradient
```

Geometrically, this looks like:

```text
move boundary a little
rotate boundary a little
change feature weight a little
check if predictions improved
repeat
```

---

## What you should watch in the simulation

At the start:

```text
prediction region is messy
loss is high
accuracy is low
```

During training:

```text
boundaries move
probability field changes
misclassified points reduce
```

After training:

```text
the decision region resembles the triangular rays
```

The network did not receive explicit triangle formulas. It learned useful boundaries from examples.
