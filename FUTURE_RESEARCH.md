# Transformer-Guided Active MRI Acquisition: Research Idea Map

This document outlines potential extensions and research directions for transformer-based active MRI acquisition using token uncertainty.  
The focus is on **deriving improved acquisition signals from the transformer's token predictions** and **improving the latent token representation itself**.

Excluded from this document:
- temperature scaling
- label smoothing
- ensembles
- MC dropout
- Dirichlet / evidential uncertainty models

---

# 1. Improving the Token Representation

## 1.1 Token Mixtures

### Motivation
Current token prediction assumes each patch corresponds to **one discrete codebook entry**.

However many patches:
- lie between codebook atoms
- correspond to boundaries
- are partially observed due to undersampling

A **mixture of tokens** allows smoother representation and better uncertainty modeling.

### Representation
Instead of selecting a single token:

```

token_embedding = codebook[token_id]

```

Use a weighted combination:

```

latent = Σ_k (w_k * codebook_embedding_k)

```

where

- `w_k ≥ 0`
- `Σ w_k = 1`

### Source of mixture weights

The simplest choice is to reuse the transformer's token probabilities:

```

w = softmax(token_logits)

```

### Practical variants

#### Full mixture
Use all token probabilities.

Pros:
- simplest
- differentiable

Cons:
- expensive with large codebooks

---

#### Top-k mixture
Keep only the top-k tokens.

```

w_k = softmax(top_k_logits)

```

Pros:
- sparse
- computationally efficient

---

#### Token + residual
Use the best token and predict a residual vector:

```

latent = codebook[token] + residual

```

Pros:
- preserves discrete interpretability
- reduces quantization error

---

### Why mixtures help acquisition

Mixtures produce richer uncertainty patterns:

- confident single-token prediction
- two-token ambiguity
- diffuse multi-token uncertainty

These patterns may provide **better acquisition signals than categorical entropy alone**.

---

# 2. Acquisition Policy Improvements

## 2.1 Expected Entropy Reduction

### Current policies

Most uncertainty-driven acquisition methods select the line with the highest current uncertainty.

Example:

```

line_score = projected_entropy

```

### Limitation

High uncertainty does **not necessarily mean** that measuring that line will reduce uncertainty.

### Proposed criterion

Select the line that maximally reduces token uncertainty:

```

Δ(a) = H(tokens | current_measurements)
- E[ H(tokens | measurements + candidate_line) ]

```

where:

- `a` = candidate line
- `H` = token entropy

Since the first term is constant across candidate lines, this reduces to minimizing the **expected future entropy**.

---

### Practical approximations

#### Plug-in reconstruction
Use the current reconstruction to simulate candidate lines.

Procedure:

1. reconstruct current image
2. simulate k-space line
3. recompute token entropy
4. rank lines by resulting entropy

---

#### Multi-hypothesis simulation
Generate several plausible reconstructions or token grids and average their entropy reduction.

---

#### Gradient-based surrogate
Use the sensitivity of token entropy to k-space:

```

score(line) = | ∂H(tokens) / ∂k_space |

```

This approximates the effect of measuring that line.

---

## 2.2 Mutual Information-Based Acquisition

### Motivation
Entropy measures total uncertainty but does not distinguish:

- reducible uncertainty
- inherent ambiguity

### Mutual information formulation

Given multiple stochastic predictions:

```

p_1, p_2, ..., p_M

```

Compute

```

predictive_entropy = H(mean(p))
expected_entropy   = mean(H(p_i))

MI = predictive_entropy - expected_entropy

```

Interpretation:

- high MI → model disagreement
- uncertainty likely reducible by more data

Use MI maps instead of entropy maps when ranking candidate lines.

---

## 2.3 Hybrid LES-GEO Policies

### Motivation

LES:
- fast
- global
- heuristic

GEO:
- targeted
- gradient-based
- slower

Combining them can yield better trade-offs.

---

### LES → GEO candidate refinement

1. LES ranks all lines
2. keep top-k lines
3. apply GEO only on these candidates

---

### Stagewise acquisition

Early steps:

```

policy = LES

```

Late steps:

```

policy = GEO

```

Rationale:

- early sampling gathers global structure
- later sampling resolves local ambiguity

---

### Weighted hybrid

```

score = λ * LES_score + (1 - λ) * GEO_score

```

with λ possibly decreasing over acquisition steps.

---

# 3. Alternative Transformer-Derived Acquisition Signals

## 3.1 Margin / Confidence Gap

### Motivation
Entropy can be dominated by many low-probability classes.

A simpler measure of ambiguity is the **gap between the top two token probabilities**.

### Definition

```

margin = p_top1 - p_top2

```

Low margin → ambiguous token identity.

Define uncertainty score:

```

uncertainty = 1 - margin

```

---

## 3.2 Top-k Mass Concentration

Instead of looking at the top two tokens, examine the mass of the top-k tokens.

```

concentration_k = Σ(top_k_probs)
uncertainty = 1 - concentration_k

```

Low concentration indicates diffuse token distributions.

---

## 3.3 Attention-Weighted Uncertainty

### Motivation

Some patches are uncertain but **not globally important** for reconstruction.

Transformer attention provides a notion of patch importance.

### Procedure

1. compute token uncertainty `u_i`
2. compute attention importance `a_i`
3. combine:

```

score_i = u_i * a_i

```

Use `score_i` instead of raw uncertainty.

---

### Possible attention measures

- mean incoming attention
- mean outgoing attention
- attention rollout across layers
- last-layer attention maps

---

# 4. Integration with Existing Policies

All signals described above can replace or augment the existing uncertainty maps used in LES or GEO.

### LES-style pipeline

```

patch_uncertainty_map
→ upsample to image resolution
→ FFT projection
→ aggregate per k-space line
→ select next line

```

### GEO-style pipeline

```

define uncertainty objective
→ compute gradient wrt k-space
→ rank lines by gradient magnitude

```

---

# 5. Recommended Experiment Roadmap

### Stage 1 (easy improvements)

- margin-based uncertainty
- top-k concentration
- hybrid LES/GEO scheduling

---

### Stage 2 (representation improvements)

- token mixtures
- attention-weighted uncertainty

---

### Stage 3 (information-theoretic policies)

- expected entropy reduction
- mutual-information acquisition

---

# 6. Conceptual Summary

Transformer-based token prediction provides a rich source of acquisition signals beyond simple entropy.

Key directions include:

- richer latent token representations
- information-theoretic acquisition policies
- uncertainty signals derived from token distributions
- structural weighting using attention

These approaches aim to guide MRI acquisition toward **reducing uncertainty over anatomical representations rather than raw pixel values**.
