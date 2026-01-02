import numpy as np

def cosine(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return np.dot(a, b) / (norm_a * norm_b)

def pearson(a, b):
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    return cosine(a_centered, b_centered)

print("="*60)
print("DEMONSTRATION: Why Raw Cosine is Misleading for LayerNorms")
print("="*60)

# 1. Simulate two UNRELATED LayerNorm vectors
# LayerNorms are initialized to 1.0. Training adds small noise (e.g., std=0.02).
dim = 4096
np.random.seed(42)

# Vector A: Mean 1.0, random noise
vec_a = 1.0 + np.random.normal(0, 0.02, dim)

# Vector B: Mean 1.0, DIFFERENT random noise (completely unrelated)
vec_b = 1.0 + np.random.normal(0, 0.02, dim)

print(f"Vector Dimension: {dim}")
print(f"Vector A: mean={np.mean(vec_a):.4f}, std={np.std(vec_a):.4f}")
print(f"Vector B: mean={np.mean(vec_b):.4f}, std={np.std(vec_b):.4f}")
print("-" * 60)

# 2. Calculate Metrics
cos_sim = cosine(vec_a, vec_b)
pearson_corr = pearson(vec_a, vec_b)

print(f"Raw Cosine Similarity:  {cos_sim:.6f}  <-- VERY HIGH (Artifact!)")
print(f"Pearson Correlation:    {pearson_corr:.6f}  <-- LOW (Correctly shows no relation)")

print("-" * 60)
print("INTERPRETATION:")
print("These two vectors are completely random and unrelated.")
print("Yet, Raw Cosine says they are 99.9% similar.")
print("This is why we cannot trust Raw Cosine for LayerNorms.")
print("Pearson correctly identifies that they share no pattern.")
