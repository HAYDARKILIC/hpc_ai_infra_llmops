# Roofline Model — Derivation & Practical Use

## 1. Setup

Let a GPU kernel perform $W$ floating-point operations while moving $Q$ bytes of data to and from HBM. Define

- **Arithmetic intensity:** $I = W / Q$ (FLOPs per byte).
- **Peak compute:** $P_{\text{peak}}$ (FLOPS).
- **Peak bandwidth:** $B_{\text{peak}}$ (B/s).

Total kernel runtime is bounded below by

$$
t \;\ge\; \max\!\left(\frac{W}{P_{\text{peak}}},\; \frac{Q}{B_{\text{peak}}}\right).
$$

Inverting, the attainable performance is

$$
\boxed{\;P_{\text{attainable}}(I) \;=\; \min\!\left(P_{\text{peak}},\; B_{\text{peak}}\cdot I\right)\;}
$$

## 2. The Ridge Point

The two regimes meet where $B_{\text{peak}} \cdot I^{*} = P_{\text{peak}}$, i.e.

$$
I^{*} \;=\; \frac{P_{\text{peak}}}{B_{\text{peak}}}.
$$

For NVIDIA H100 at BF16: $I^{*} \approx 989 \text{ TFLOPS} / 3.35 \text{ TB/s} \approx 295$ FLOPs/byte.

A kernel is **memory-bound** iff $I < I^{*}$.

## 3. Practical Classification

For a matmul $C = AB$ with $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$:

- Work: $W = 2 M N K$ FLOPs.
- Bytes (BF16): $Q = 2(MK + KN + MN)$.
- Intensity: $I = \dfrac{2MNK}{2(MK + KN + MN)} = \dfrac{MNK}{MK + KN + MN}$.

For a square matmul $M = N = K = d$: $I = d/3$. So at $d \ge 3 I^{*}$ the matmul is compute-bound on H100 BF16 — which sets a useful lower bound on layer width.

## 4. Why Lower Precision Helps Memory-Bound Kernels

Halving the dtype size halves $Q$ but leaves $W$ unchanged, so $I$ *doubles*. A kernel that was memory-bound at BF16 may become compute-bound at FP8 — and the FP8 compute peak is also higher, compounding the win.
