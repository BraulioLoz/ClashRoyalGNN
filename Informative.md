# Clash Royale GNN - Loss Function Deep Dive

This document provides a comprehensive explanation of the loss function used in our card recommendation system, including the win/loss weighted variant.

---

## Table of Contents

1. [Problem Context](#problem-context)
2. [Loss Function Overview](#loss-function-overview)
3. [Step-by-Step Computation](#step-by-step-computation)
4. [Win/Loss Weighted Loss](#winloss-weighted-loss)
5. [Mathematical Formulation](#mathematical-formulation)
6. [Implementation Details](#implementation-details)
7. [Gradient Flow Analysis](#gradient-flow-analysis)

---

## Problem Context

### The Task

Given 6 cards from a player's deck, predict the 2 remaining cards that complete the deck.

```
Input (X):  [Card_1, Card_2, Card_3, Card_4, Card_5, Card_6]  → 6 known cards
Output:     P(card | X) for all ~110 cards                    → probability distribution
Target (Y): [Card_7, Card_8]                                  → 2 cards to predict
```

### Why Standard Cross-Entropy Doesn't Work Directly

Standard multi-class cross-entropy assumes:
- Single correct class
- All classes are valid candidates

Our problem has constraints:
- **Multiple correct answers** (2 target cards)
- **Invalid candidates** (the 6 input cards cannot be predicted)

This requires a **modified cross-entropy with masking**.

---

## Loss Function Overview

The loss function is implemented in `src/pipelines/training_pipeline.py:compute_loss()`.

### High-Level Flow

```
GNN Output (logits)
        ↓
   Aggregate across nodes (mean/max)
        ↓
   Mask input cards (set to -∞)
        ↓
   Softmax → Probabilities
        ↓
   Cross-entropy with target distribution
        ↓
   Apply sample weight (for win/loss)
        ↓
   Final weighted loss
```

---

## Step-by-Step Computation

### Step 1: Aggregate Node Logits

The GNN produces logits for each node in the graph. We aggregate these to get a single prediction vector.

```python
# logits shape: [num_nodes, num_cards] = [110, 110]
# Each node produces predictions for all cards

if loss_aggregation == "mean":
    node_logits = logits.mean(dim=0)  # [110]
elif loss_aggregation == "max":
    node_logits = logits.max(dim=0)[0]  # [110]
```

**Why aggregate?** The GNN processes the entire card graph. Each node (card) produces predictions, but we need a single distribution over cards. Mean aggregation treats all nodes equally; max aggregation emphasizes confident predictions.

### Step 2: Mask Input Cards

The 6 input cards cannot be recommended (they're already in the deck). We exclude them by setting their logits to negative infinity.

```python
# Create mask: True for valid cards, False for input cards
mask = torch.ones(num_cards, dtype=torch.bool)  # [110]
mask[input_card_indices] = False                 # Mark 6 input cards as invalid

# Apply mask using torch.where
masked_logits = torch.where(
    mask,
    node_logits,                    # Keep original logit if valid
    torch.tensor(float('-inf'))     # Set to -∞ if input card
)
```

**Why -∞?** After softmax, \( e^{-\infty} = 0 \), so input cards get exactly 0 probability. This is numerically stable and ensures the model cannot "cheat" by predicting known cards.

### Step 3: Create Target Distribution

We convert the 2 target cards into a probability distribution.

```python
# target_indices = [idx_card7, idx_card8] (the 2 target cards)

target = torch.zeros(num_cards)  # [110]
target[target_indices] = 1.0     # Set 1 at target positions

# Normalize to sum to 1
target = target / target.sum()   # Each target gets 0.5
```

**Result**: If cards 42 and 87 are targets:
```
target = [0, 0, ..., 0.5, ..., 0.5, ..., 0]
                      ↑ idx 42   ↑ idx 87
```

This is a **soft label** distribution where both target cards are equally important.

### Step 4: Apply Softmax

Convert masked logits to probabilities.

```python
probs = torch.softmax(masked_logits, dim=0)  # [110]
```

**Softmax formula**:

$$
P(i) = \frac{e^{z_i}}{\sum_{j \in \text{valid}} e^{z_j}}
$$

Where \( z_i \) is the logit for card \( i \), and the sum is only over valid (non-input) cards.

**Key property**: Input cards have \( z_i = -\infty \), so \( e^{-\infty} = 0 \), and they contribute nothing to the denominator.

### Step 5: Compute Cross-Entropy

Measure how well the predicted distribution matches the target distribution.

```python
loss = -torch.sum(target * torch.log(probs + 1e-8))
```

**Expanded**:

$$
\mathcal{L}_{\text{base}} = -\sum_{i=1}^{N} y_i \cdot \log(P(i) + \epsilon)
$$

Where:
- \( y_i \) = target probability for card \( i \) (0.5 for targets, 0 otherwise)
- \( P(i) \) = predicted probability for card \( i \)
- \( \epsilon = 10^{-8} \) = small constant for numerical stability

**For 2 target cards (indices \( t_1, t_2 \))**:

$$
\mathcal{L}_{\text{base}} = -\frac{1}{2}\log P(t_1) - \frac{1}{2}\log P(t_2)
$$

This is the **average negative log-likelihood** of the target cards.

### Step 6: Apply Sample Weight (Win/Loss Weighting)

Finally, apply the sample weight based on battle outcome.

```python
weighted_loss = loss * sample_weight
```

---

## Win/Loss Weighted Loss

### Motivation

Not all training examples are equally valuable:
- **Winning decks**: Provide positive signal about effective card combinations
- **Losing decks**: May indicate poor synergies (less useful signal)
- **Dominant wins (3-crown)**: Strong evidence of deck effectiveness
- **Close losses (1-crown)**: Still somewhat informative

### Weight Configuration

Configured in `config/config.yaml`:

```yaml
training:
  margin_weights:
    3: 1.0    # 3-crown win (dominant victory)
    2: 0.8    # 2-crown win
    1: 0.6    # 1-crown win
    0: 0.0    # Draw (skip - no signal)
    -1: 0.2   # 1-crown loss
    -2: 0.1   # 2-crown loss
    -3: 0.05  # 3-crown loss (least useful)
```

### Crown Margin Calculation

From battle logs:

```python
crown_margin = team_crowns - opponent_crowns  # Range: -3 to +3
```

| Crown Margin | Meaning | Weight | Relative Influence |
|--------------|---------|--------|-------------------|
| +3 | 3-crown win | 1.0 | 20x vs 3-crown loss |
| +2 | 2-crown win | 0.8 | 16x vs 3-crown loss |
| +1 | 1-crown win | 0.6 | 12x vs 3-crown loss |
| 0 | Draw | 0.0 | Skipped |
| -1 | 1-crown loss | 0.2 | 4x vs 3-crown loss |
| -2 | 2-crown loss | 0.1 | 2x vs 3-crown loss |
| -3 | 3-crown loss | 0.05 | Baseline |

### Weighted Loss Formula

$$
\mathcal{L}_{\text{weighted}} = w \cdot \mathcal{L}_{\text{base}}
$$

Where \( w \) is the sample weight based on crown margin.

### Batch Normalization

When processing a batch, we normalize by total weight instead of batch size:

```python
# Standard: batch_loss = sum(losses) / batch_size
# Weighted: batch_loss = sum(weighted_losses) / sum(weights)

batch_loss = 0.0
total_weight = 0.0

for i in range(batch_size):
    loss = compute_loss(..., sample_weight=weights[i])
    batch_loss += loss
    total_weight += weights[i]

batch_loss = batch_loss / total_weight
```

This ensures that the gradient magnitude is consistent regardless of the weight distribution in the batch.

---

## Mathematical Formulation

### Complete Loss Function

For a single training example:

$$
\mathcal{L}(X, Y, w) = -w \cdot \sum_{i \in Y} \frac{1}{|Y|} \log \frac{e^{z_i}}{\sum_{j \notin X} e^{z_j}}
$$

Where:
- \( X \) = set of 6 input card indices
- \( Y \) = set of 2 target card indices
- \( w \) = sample weight (from crown margin)
- \( z_i \) = aggregated logit for card \( i \)

### Expanded Form

$$
\mathcal{L} = -\frac{w}{2} \left[ \log P(t_1 | X) + \log P(t_2 | X) \right]
$$

$$
= -\frac{w}{2} \left[ z_{t_1} - \log\sum_{j \notin X} e^{z_j} + z_{t_2} - \log\sum_{j \notin X} e^{z_j} \right]
$$

$$
= -\frac{w}{2} \left[ z_{t_1} + z_{t_2} - 2\log\sum_{j \notin X} e^{z_j} \right]
$$

### Gradient with Respect to Logits

For a target card \( t \):

$$
\frac{\partial \mathcal{L}}{\partial z_t} = w \cdot \left( P(t|X) - \frac{1}{|Y|} \right)
$$

For a non-target, non-input card \( k \):

$$
\frac{\partial \mathcal{L}}{\partial z_k} = w \cdot P(k|X)
$$

**Interpretation**:
- **Target cards**: Gradient pushes logits UP (to increase \( P(t) \) toward 0.5)
- **Non-target cards**: Gradient pushes logits DOWN (to decrease \( P(k) \) toward 0)
- **Input cards**: No gradient (masked out)
- **Weight \( w \)**: Scales gradient magnitude (winning decks get stronger updates)

---

## Implementation Details

### Code Location

`src/pipelines/training_pipeline.py`

### Key Functions

| Function | Purpose |
|----------|---------|
| `compute_loss()` | Computes weighted loss for single example |
| `process_batch_examples()` | Processes batch with weights |
| `create_input_mask()` | Creates boolean mask for input cards |
| `collate_fn()` | Batches data, targets, and weights |

### Numerical Stability

1. **Log stability**: `log(probs + 1e-8)` prevents `log(0)`
2. **Softmax stability**: PyTorch's softmax is numerically stable
3. **Masking**: Using `-inf` instead of large negative numbers ensures exact 0 probability

### Memory Efficiency

- Masks created vectorized (no Python loops)
- Targets created vectorized with index assignment
- Batch processing on GPU with non-blocking transfers

---

## Gradient Flow Analysis

### What the Model Learns

1. **From winning decks (high weight)**:
   - Strong signal about which cards work well together
   - Model learns "these 2 cards complete winning combinations"

2. **From losing decks (low weight)**:
   - Weak signal (might be bad matchup, not bad cards)
   - Model still learns co-occurrence but with less confidence

3. **From skipped draws (weight=0)**:
   - No gradient contribution
   - Draws provide ambiguous signal (neither won nor lost)

### Effect on Training Dynamics

| Scenario | Effect |
|----------|--------|
| Batch of all wins | Full gradient, fast learning |
| Batch of all losses | Reduced gradient, cautious learning |
| Mixed batch | Weighted average, balanced learning |

### Convergence Behavior

With win/loss weighting:
- **Faster convergence** on winning patterns
- **Slower forgetting** of losing patterns
- **Bias toward winning deck compositions**

This is intentional: we want the model to recommend cards that **maximize win probability**, not just cards that frequently appear together.

---

## Summary

The loss function combines:

1. **Cross-entropy** for multi-label classification
2. **Input masking** to exclude known cards
3. **Target normalization** for multiple correct answers
4. **Sample weighting** to prioritize winning decks

**Final formula**:

$$
\boxed{\mathcal{L} = -w \cdot \sum_{t \in \text{targets}} \frac{1}{|\text{targets}|} \log P(t | \text{input cards masked})}
$$

Where \( w \in [0.05, 1.0] \) based on battle outcome (crown margin).
