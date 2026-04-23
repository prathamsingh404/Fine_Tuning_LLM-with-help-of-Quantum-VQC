# Quantum vs Classical Layer Comparison

Project: `Quantum Mechanics and Quantum Computing`

This report documents the behavior of the quantum and classical branches in this project, explains how the two layers are constructed, and summarizes the observed training results from the actual run recorded in this repository. The goal is to compare both branches fairly, using the same outer transformer pipeline and the same bottleneck size, so that the only meaningful difference is the quantum circuit versus the classical nonlinear replacement.

## 1. Executive Summary

The project compares two sentiment-analysis models:

- `QuantumTransformer`
- `ClassicalTransformer`

Both models share the same overall architecture:

- token embedding
- positional encoding
- two Transformer encoder blocks
- a post-encoder bottleneck block
- global average pooling
- linear classification head

The difference is in the post-encoder bottleneck:

- the quantum model uses a Variational Quantum Circuit (VQC)
- the classical model uses a matched classical bottleneck of the same input/output shape

In the current version of the code, both branches use the same bottleneck width of `10`:

- `64 -> 10 -> 64` in the quantum branch
- `64 -> 10 -> 64` in the classical branch

That is important because it removes the most obvious “capacity mismatch” excuse. The two branches are now structurally comparable in size and dimension. The observed result from training, however, is that the classical branch learns substantially faster and achieves much better validation accuracy on SST-2.

Observed best validation accuracy from the recorded run:

- Quantum model: `0.5092`
- Classical model: `0.8200`

This means:

- the quantum model performed only slightly above random guessing on a binary task
- the classical model learned a useful sentiment classifier
- the quantum bottleneck, as implemented here, did not improve accuracy in this experiment

## 2. What “Same Parameters” Means Here

When people say “same parameters” in a fair comparison, there are two common meanings:

1. Same outer architecture and same training pipeline
2. Similar parameter budget or identical bottleneck dimensions

This project satisfies the first meaning strongly and the second meaning reasonably well.

The two models both use:

- the same vocabulary size
- the same embedding size (`64`)
- the same number of Transformer encoder blocks (`2`)
- the same classifier head
- the same dataset (`SST-2`)
- the same training objective
- the same evaluation procedure
- the same bottleneck width (`10`)

The actual quantum and classical branches differ only in the replacement module:

- quantum: a 10-qubit VQC
- classical: a small feed-forward bottleneck with the same `10`-dimensional internal width

So this is a fair comparison of two different nonlinear “middle blocks,” not a comparison of a large model versus a tiny model.

## 3. Architecture Overview

### 3.1 Shared Backbone

Both models start with the same backbone:

1. Tokenize input text into integer IDs
2. Convert IDs into learned embeddings
3. Add learned positional encodings
4. Pass through two `TransformerEncoderLayer` blocks
5. Apply the final branch-specific attention/bottleneck layer
6. Mean-pool across sequence length
7. Classify into `Positive` or `Negative`

This backbone is important because it means the VQC is not doing all the heavy lifting. The transformer body already extracts contextual structure from the input. The branch-specific layer acts more like a nonlinear refinement stage after the classical representation is formed.

### 3.2 Quantum Branch

The quantum branch uses `QuantumAttentionLayer`, which works like this:

- project 64-dimensional token features down to 10 values
- apply `tanh` to constrain them to a bounded range
- encode them into a 10-qubit circuit
- apply a parameterized quantum circuit with one variational layer
- measure 10 expectation values
- project the 10 outputs back to 64 dimensions

In formula form:

`x -> Linear(64, 10) -> Tanh -> VQC(10 qubits) -> Linear(10, 64)`

The circuit itself uses:

- `RY` data encoding
- a CNOT entangling ring
- trainable `RY` and `RZ` rotations
- another CNOT ring
- Pauli-Z measurements on every qubit

### 3.3 Classical Branch

The classical branch now mirrors the same bottleneck shape:

`x -> Linear(64, 10) -> Tanh -> Linear(10, 64)`

This is a very important design choice. It means:

- both branches compress to 10 internal features
- both branches expand back to 64 features
- both branches preserve the same outer tensor shape

So the comparison is no longer “quantum circuit versus one linear layer.” It is “quantum circuit versus a classical nonlinear bottleneck of the same width.”

## 4. Why the Quantum Layer Was Originally Harder to Train

Before the bottleneck was widened to `10`, the quantum layer used a narrower 8-qubit circuit. That setup made the quantum side more constrained than the classical side and could have unintentionally reduced its learning capacity.

Even after matching the width, the quantum layer still has several training disadvantages:

### 4.1 Circuit evaluation cost

The quantum layer is much more expensive to evaluate than a classical linear layer. Each forward pass through the VQC requires circuit simulation. Each backward pass can also be expensive because gradients need to be computed through the circuit.

### 4.2 More fragile optimization landscape

Variational quantum circuits often have rugged optimization surfaces. The model may get stuck in shallow regions or fail to find strong improvements without careful circuit design and hyperparameter tuning.

### 4.3 Information bottleneck

The data is compressed into 10 quantum inputs before the circuit. That compression is shared with the classical model, but the quantum branch has to learn a useful representation while also respecting the circuit’s output constraints.

### 4.4 Limited circuit depth

The circuit currently has:

- 10 qubits
- 1 variational layer
- one fixed entangling pattern before and after the variational block

This is not a very expressive circuit for a real NLP benchmark. It is enough to demonstrate hybrid quantum computation, but it may not be enough to outperform a classical bottleneck on a task like SST-2.

## 5. Training Setup

The recorded training run used:

- `SST-2` from the GLUE benchmark
- `67,349` training samples
- `872` validation samples
- vocabulary size `10,000`
- sequence length `128`
- embedding dimension `64`
- quantum bottleneck width `10`
- classical bottleneck width `10`

Training schedule:

- quantum model: `5` epochs
- classical model: `10` epochs

This is a notable asymmetry in the training schedule. Even though both branches have the same architecture size, the classical model was trained for twice as many epochs. That makes the classical result stronger, but it also means the comparison is not perfectly balanced in terms of optimization time.

Still, the quantum model’s curve was flat enough that even a modest increase in epochs would not obviously close the gap.

## 6. Observed Results

### 6.1 Quantum Model

Training behavior:

- training loss stayed around `0.689`
- training accuracy stayed around `0.545`
- validation accuracy hovered around `0.509`

Best validation accuracy:

- `0.5091743119266054`

Training time:

- `2560.19` seconds

Parameters:

- `716,378`

Interpretation:

- the quantum model barely learned beyond chance
- the loss stayed close to binary-entropy for an uncertain classifier
- training was slow
- the model did not develop a strong separating boundary on validation data

### 6.2 Classical Model

Training behavior:

- training loss dropped from `0.5283` to `0.1541`
- training accuracy rose from `0.7185` to `0.9336`
- validation accuracy peaked at `0.8200`

Best validation accuracy:

- `0.819954128440367`

Training time:

- `2947.32` seconds

Parameters:

- `716,620`

Interpretation:

- the classical model learned effectively
- it fit the training distribution much better
- it generalized well enough to reach `~82%` validation accuracy
- despite being only slightly larger, it dramatically outperformed the quantum branch

## 7. Quantitative Comparison

### 7.1 Best Validation Accuracy

- Quantum: `50.92%`
- Classical: `82.00%`

Difference:

- about `31.08 percentage points`

This is a large gap. In practical terms, the classical layer is clearly superior in this implementation.

### 7.2 Training Loss Trend

Quantum model:

- almost flat
- loss values clustered near `0.69`
- indicates weak learning signal or poor optimization progress

Classical model:

- steadily decreasing loss
- strong and consistent learning
- clear separation between training and validation behavior

### 7.3 Overfitting Signal

The classical model does show some overfitting:

- training accuracy keeps rising
- validation accuracy peaks around epoch 4 and then fluctuates

But this is a healthy and expected pattern in a successful model. It learned useful features before validation performance saturated.

The quantum model does not show meaningful overfitting. Instead, it shows underfitting:

- it fails to improve much on either train or validation data

That tells us the issue is not “it generalizes too badly after learning too much.”
It is more like “it never learned enough in the first place.”

## 8. Why the Quantum Layer Likely Underperformed

There is no single reason. The most likely explanation is a combination of design and optimization limits.

### 8.1 The circuit is too shallow

One variational layer is very small. It is a legitimate demonstration, but not a strong classifier by itself. More layers could increase expressiveness.

### 8.2 The model only sees a compressed view

The input to the VQC is a learned projection into 10 dimensions. That is a very narrow information bottleneck. If the transformer features are not already highly discriminative, the circuit may not have enough room to recover useful structure.

### 8.3 Quantum outputs are bounded

The circuit outputs expectation values in a bounded range. That is useful for stability, but it can also limit the scale of features flowing onward.

### 8.4 Subsampling reduces coverage

The quantum branch processes only a subset of sequence positions per stride and interpolates the rest. That makes the layer faster, but it also means not every token gets direct circuit processing.

### 8.5 Optimization noise and circuit complexity

VQC training can be sensitive to initialization, gradient path quality, and circuit architecture. A classical linear layer is much easier to optimize.

### 8.6 The benchmark favors strong classical baselines

SST-2 is a standard sentiment task. Classical transformer-like models are very strong here, and a small hybrid quantum layer has to overcome a high bar.

## 9. What the “10 Apples” Analogy Means

Your intuition is good.

When you said both should have “10 apples,” the practical interpretation is:

- both branches should compress into the same small internal representation
- both branches should get the same amount of intermediate capacity
- both branches should then expand back to the same output size

That is exactly what the current design does:

- quantum branch: `64 -> 10 -> 64`
- classical branch: `64 -> 10 -> 64`

This matters because if one branch had `64 -> 8 -> 64` and the other had `64 -> 64 -> 64`, the comparison would be unfair. The wider branch would simply have more capacity and could win for reasons unrelated to the quantum/classical difference.

## 10. What We Can Conclude

From the current experiment, the conclusion is:

1. The quantum layer is a valid hybrid module and the pipeline works end-to-end.
2. The classical layer is much easier to optimize and performs much better on this task.
3. Matching the bottleneck size makes the comparison fairer, but it does not automatically make the quantum model better.
4. The current quantum circuit is probably too small and too constrained to beat the classical bottleneck on SST-2.

So the core experimental observation is not “quantum beats classical.”
It is the opposite:

- the classical bottleneck is significantly stronger in this setup
- the quantum bottleneck is functioning, but it is not yet competitive

## 11. Practical Interpretation for the Report

If you are writing this up for your project, a balanced and academically honest framing would be:

- the quantum circuit was integrated successfully into a transformer-like NLP pipeline
- both branches were matched to the same bottleneck size for a fair comparison
- the classical baseline achieved substantially higher validation accuracy
- the quantum model remained close to chance and did not show meaningful improvement
- this suggests that, for this task and this circuit design, the quantum layer is not yet expressive enough or easy enough to optimize to outperform the classical alternative

That is a strong and respectable result for a project report because it demonstrates:

- implementation success
- fair experimental design
- honest evaluation
- clear evidence-based comparison

## 12. Suggested Future Improvements

If you want to improve the quantum branch in future work, the most promising directions are:

1. Increase circuit depth
2. Add data re-uploading
3. Try a larger or richer entanglement pattern
4. Tune the stride so more tokens are processed directly
5. Train the quantum branch with more careful learning-rate schedules
6. Add residual connections around the quantum layer
7. Run multiple random seeds and average results

If you want the report to sound more research-like, you can say:

- “The current quantum circuit serves as a proof of concept rather than a state-of-the-art classifier.”
- “Performance appears limited by circuit expressiveness and optimization difficulty.”
- “The classical bottleneck provided a stronger inductive bias for this benchmark under identical outer model conditions.”

## 13. Final Takeaway

The final story of this project is clear:

- both branches are now matched in size
- both branches process the same transformer features
- both branches have the same `64 -> 10 -> 64` internal shape
- the classical bottleneck learns the task well
- the quantum bottleneck is functional but currently underpowered for this benchmark

In other words, the experiment shows that a quantum layer can be integrated into an NLP pipeline, but in this implementation it does not yet outperform a comparable classical bottleneck.

