# CerebroSpheroid V6: Research Notes & Architectural Specifications

**Document Version:** 6.0.0  
**Date:** November 17, 2026  
**Status:** Living Architectural Specification & Research Log  

---

## 1. The Radially-Tapered Cortical Spheroid (Quadratic Geometric Scaling)

### 1.1 The Core Geometric Realization: Node Density Flows Outward
In standard neural architectures (and in CerebroSpheroid V5), all recurrent laminae or layers are assigned a uniform channel width $D$ (e.g. $D=640$ or $D=1024$). While uniform dimensions simplify tensor shapes, they fundamentally contradict the physics and geometry of spherical cavities.

In 3D Euclidean space, the surface area $A(r)$ and differential volume $dV(r)$ of a concentric spherical shell at normalized radius $r \in (0, 1]$ scale quadratically:
$$A(r) = 4\pi r^2, \quad dV(r) = 4\pi r^2 \, dr$$

If neuron or feature density is distributed volumetrically, the capacity of any radial shell is strictly a quadratic function of radius $r$. 
- The outermost 20% of the radius ($r \in [0.8, 1.0]$) contains **48.8%** of the total volume.
- The deep core ($r \in [0.0, 0.2]$) contains only **0.8%** of the volume.

This mirrors the evolutionary anatomy of the mammalian brain: the outer cerebral cortex is an ultra-dense, folded sheet with 16 billion neurons, while the subcortical core (thalamus, basal ganglia) occupies minimal volume and functions as a low-dimensional orchestrator.

---

### 1.2 Mathematical Formulation of Cortical Tapering

In CerebroSpheroid V6, the channel dimension $D_k$ of shell $k \in \{0, 1, \dots, K-1\}$ is defined by quadratic radial interpolation:

$$D_k = D_{\text{core}} + (D_{\text{mantle}} - D_{\text{core}}) \cdot \left(\frac{k}{K-1}\right)^2$$

Where:
- $k = 0$ is the deep subcortical core ($D_0 = D_{\text{core}}$).
- $k = K - 1$ is the outer cortical surface ($D_{K-1} = D_{\text{mantle}}$).
- All $D_k$ are rounded to the nearest multiple of 16 (or 32) to ensure strict alignment with TPU MXU systolic arrays and GPU Tensor Cores.

#### Tapering Progression Example ($K = 48, D_{\text{core}} = 256, D_{\text{mantle}} = 1536$):
- Shells $0 \dots 7$ (Deep Subcortical Core): $D_k \in [256, 288]$
- Shells $8 \dots 23$ (Intermediate Subcortical / Relational): $D_k \in [320, 640]$
- Shells $24 \dots 39$ (Association Cortex): $D_k \in [704, 1152]$
- Shells $40 \dots 47$ (Outer Mantle / Sensory-Motor Surface): $D_k \in [1216, 1536]$

---

## 2. Theoretical Solutions to the V5 Scaling Plateau

### 2.1 Resolving the Information Bottleneck & Diminishing Returns
In CerebroSpheroid V5, scaling shell depth beyond $K = 32$ at fixed $D = 640$ yielded diminishing returns. The empirical and theoretical root causes were:
1. **Core Over-Parameterization:** Deep shells ($k < 10$) possessed 640 dimensions when they only needed to track compact phase clocks and semantic centroids, wasting parameter budget and computation.
2. **Sensory Channel Bandwidth Starvation:** By the Data Processing Inequality, mutual information $I(X; S_{\text{core}}) \le I(X; S_{\text{surface}})$. At $D = 640$, the input channel capacity was too narrow to populate 48 deep shells with non-redundant representations.

**The V6 Resolution:**
By expanding the outer mantle to $D_{\text{mantle}} = 1536 \dots 2048$, the sensory ingestion bandwidth is expanded by $2.4\times$ to $3.2\times$. Simultaneously, tapering the core down to $D_{\text{core}} = 256$ prevents over-parameterization, allowing $K$ to scale to **$K = 48 \dots 64$ shells** without hitting the information-theoretic plateau.

---

### 2.2 Bidirectional Wave Dynamics: Centripetal vs. Centrifugal Flow

```
                      INWARD (Centripetal)
                     [Abstraction & Invariant Extraction]
              D_mantle = 1536  ──►  D_mid = 768  ──►  D_core = 256
             ┌────────────────────────────────────────────────────┐
  SENSORY    │ Shell 47 (1536)                                    │
  INPUTS ───►│   Shell 32 (1024)                                  │
             │     Shell 16 (512)                                 │
             │       Shell 0: DEEP CORE (256) [Concept Attractor] │
             │     Shell 16 (512)                                 │
  MOTOR  ◄───│   Shell 32 (1024)                                  │
  LOGITS     │ Shell 47 (1536)                                    │
             └────────────────────────────────────────────────────┘
                     OUTWARD (Centrifugal)
                   [Decompression & Articulation]
              D_core = 256  ──►  D_mid = 768  ──►  D_mantle = 1536
```

1. **Centripetal Wave (Surface $\to$ Core): Dimensional Compression**
   - Inward projections $W_{k \to k-1}^{\text{in}} \in \mathbb{R}^{D_k \times D_{k-1}}$ act as adaptive dimensionality-reduction operators.
   - High-frequency lexical, syntactic, and superficial prompt details are stripped away, distilling the prompt into an invariant semantic centroid vector in $\mathbb{R}^{D_{\text{core}}}$.

2. **Centrifugal Wave (Core $\to$ Surface): Dimensional Expansion**
   - Outward projections $W_{k \to k+1}^{\text{out}} \in \mathbb{R}^{D_k \times D_{k+1}}$ decompress the core invariant.
   - As the signal re-emerges into higher-dimensional outer shells, it combines with the Arcuate Fasciculus cross-resonance KV cache to predict the exact next token with rich grammatical and tonal precision.

---

## 3. Computational and Hardware Economics

### 3.1 Matrix Multiplication Latency Advantages
In standard uniform models, every layer operation costs $O(D_{\text{mantle}}^2)$. In a tapered spheroid:
- Core operations: $256 \times 256 = 65,536$ FLOPs.
- Mantle operations: $1536 \times 1536 = 2,359,296$ FLOPs ($36\times$ more compute).

The average shell FLOP cost across the tapered hierarchy is:
$$\bar{\text{FLOPs}} \approx \frac{1}{K} \sum_{k=0}^{K-1} D_k^2 \approx 0.38 \cdot D_{\text{mantle}}^2$$

**Result:** A 64-shell tapered model runs with the computational FLOP footprint of a **24-shell uniform model** of dimension $D_{\text{mantle}}$, while delivering the algorithmic depth of **64 physical shells** (or **512 effective layer operations** at $T = 4$ thinking cycles).

---

## 4. CerebroSpheroid V6 Architecture Configurations

| Configuration | $K$ (Shells) | $D_{\text{mantle}}$ | $D_{\text{core}}$ | $D_r$ (Radial) | Heads ($H$) | Tracts | Effective Depth ($T=4$) | Target Params | Hardware Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V6_Compact** | **32** | 1024 | 256 | 384 | 8 | 96 | 256 ops | **~180M** | 1× TPU v5e / 8GB GPU |
| **V6_Standard** | **48** | 1536 | 256 | 512 | 16 | 160 | 384 ops | **~420M** | 1× TPU v6e / 16GB GPU |
| **V6_Deep** | **64** | 2048 | 384 | 768 | 16 | 224 | 512 ops | **~880M** | 2×-4× TPU v6e |
| **V6_Frontier** | **96** | 3072 | 512 | 1024 | 24 | 384 | 768 ops | **~2.4B** | 4×-8× TPU v6e Pod |

---

## 5. Architectural Invariants Preserved in V6
1. **Polar Newton-Schulz NS-5 Orthogonalization**: All inward/outward radial matrices and axonal tracts must remain clamped to spectral radius $\rho \le 1.0$.
2. **Small-World Shortcut Topology**: Axonal tracts must retain the Watts-Strogatz $\approx 10\%$ connectivity ratio with dyadic shortcut horizons ($2^p$ shell jumps).
3. **Subconscious Deliberation & Recirculation**: Thinking cycles $T_{\text{delib}} \in [1, 6]$ operate as dynamic cognitive multipliers with 0 parameter cost.
4. **Prompt Refractory Inhibition**: Anti-parroting memory clamps prevent rote echoing and compel original deductive reasoning.
