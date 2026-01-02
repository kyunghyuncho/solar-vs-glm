# Solar-Open-100B vs GLM-4.5-Air: Weight Derivation Analysis

## Final Conclusion: NO Evidence of Derivation

**Verdict: Independent Models (or completely retrained)**

After correcting for statistical artifacts in LayerNorm comparison, we find **no evidence** that Solar-Open-100B is derived from GLM-4.5-Air.

---

## The "Cosine Similarity Trap" (Why previous analysis was wrong)

Initial analysis suggested "Definitive Evidence" with Cosine Similarity scores of ~0.99 for LayerNorm weights. However, this was proven to be a statistical artifact.

### The Mathematical Proof
LayerNorm weights are initialized with a mean of 1.0 and a small variance.
- Vector A: Random noise + 1.0
- Vector B: Random noise + 1.0

Because both vectors are centered far from zero (relative to their variance), the angle between them is very small, even if the noise is completely uncorrelated.

**Demonstration (`demo_cosine_artifact.py`):**
```python
# Two RANDOM vectors with mean=1.0
vec_a = np.random.normal(1.0, 0.01, 4096)
vec_b = np.random.normal(1.0, 0.01, 4096)

Cosine Similarity: 0.9999  <-- Misleading "High Similarity"
Pearson Correlation: 0.0012 <-- Correct "No Correlation"
```

### The Correct Metric: Pearson Correlation
Pearson correlation centers the data (subtracts the mean) before calculating similarity. This removes the "mean shift" artifact and measures whether the *shapes* of the weight vectors are actually similar.

---

## Corrected Analysis Results

We re-ran the analysis using Pearson Correlation as the "Gold Standard" metric.

### 1. LayerNorm Comparison (`probe_comprehensive.py`)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Cosine Similarity** | **0.989** | **Artifact** (Due to mean ≈ 1.0) |
| **Pearson Correlation** | **0.012** | **NO RELATIONSHIP** |
| Control Baseline (Random) | 0.000 | Indistinguishable from random |

**Result:** The weights of Solar-Open-100B and GLM-4.5-Air are statistically independent. The high cosine score was purely due to the initialization properties of LayerNorms.

### 2. Layer Decay Analysis (`probe_layer_decay.py`)

If Solar was continually pretrained from GLM, we would expect:
1.  High Pearson correlation in early layers.
2.  Decaying correlation in later layers.

**Observed:**
- Pearson correlation is ~0.0 across ALL layers.
- No decay pattern exists.

### 3. Embedding Comparison (`compare_embeddings.py`)

- **Cosine:** ~0.002
- **Pearson:** ~0.002
- **Conclusion:** Embeddings are completely different.

---

## Methodology

The repository contains the following scripts used for this analysis:

### Core Scripts
- **`probe_comprehensive.py`**: The main analysis script. Calculates both Cosine and Pearson metrics.
- **`probe_layer_decay.py`**: Checks for layer-wise drift patterns.
- **`probe_with_control.py`**: Compares Solar vs GLM against a random control model to ensure statistical significance.

### Validation Scripts
- **`demo_cosine_artifact.py`**: A standalone proof demonstrating why Cosine Similarity fails for LayerNorms.
- **`definitive_proof.py`**: A simplified checker that warns if Cosine is high but Pearson is low.

---

## Summary

While Solar-Open-100B and GLM-4.5-Air share similar architectures (Llama-style), there is **zero statistical evidence** in the weights to suggest that one was initialized from the other. The earlier positive result was a false positive caused by the "Cosine Artifact" on LayerNorm weights.
