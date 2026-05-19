# Pipeline Bubble Fraction — Derivation

## 1. Setup

A model with $L$ layers is partitioned into $p$ contiguous **stages**, one per device. A macrobatch is split into $m$ **microbatches** that flow through the pipeline. Let $\tau$ be the per-microbatch per-stage time (assumed identical for forward and backward for simplicity).

## 2. GPipe schedule

GPipe issues all $m$ forward passes before any backward begins.

Total wallclock:

$$
T_{\text{GPipe}} \;=\; \underbrace{(p - 1)\tau}_{\text{fill}} + \underbrace{m\tau}_{\text{forward steady-state}} + \underbrace{(p - 1)\tau}_{\text{drain into backward}} + \underbrace{m\tau}_{\text{backward steady-state}} + (p - 1)\tau.
$$

The bubble-free ideal is $T_{\text{ideal}} = 2 m \tau$ (each microbatch costs $2\tau$ on its stage). The bubble fraction is

$$
\rho_{\text{bubble}} \;=\; 1 - \frac{T_{\text{ideal}}}{T_{\text{GPipe}}} \;=\; \frac{p - 1}{m + p - 1} \;\approx\; \frac{p - 1}{m} \quad (m \gg p).
$$

## 3. The 1F1B schedule

In **One-Forward-One-Backward (1F1B)**, once a stage's pipeline is filled, it alternates one forward microbatch and one backward microbatch. The total time is unchanged

$$
T_{\text{1F1B}} \;=\; T_{\text{GPipe}},
$$

so the bubble fraction is the same. However the **activation memory** is dramatically different.

### Activation memory: GPipe vs 1F1B

In GPipe, every microbatch's activations must be retained at every stage until its backward runs, giving per-stage memory $O(m)$.

In 1F1B, after the warmup phase a stage holds at most $p$ activations in flight at any time (one per microbatch currently between this stage's forward and its backward). Per-stage memory is $O(p)$ — **independent of $m$**.

This is the whole motivation for 1F1B: same bubble, $m/p$-fold less activation memory.

## 4. Interleaved 1F1B

If each stage owns *multiple non-contiguous chunks* of layers, the bubble shrinks further. With $v$ chunks per stage, the bubble fraction becomes

$$
\rho_{\text{interleaved}} \;\approx\; \frac{p - 1}{m \cdot v},
$$

at the cost of $v$ extra inter-stage sends per microbatch. Megatron-LM defaults to $v = 2$ or $4$.

## 5. Practical guidance

- Choose $m \ge 4 p$ to keep the bubble under 25%.
- 1F1B is strictly better than GPipe in memory, never worse in compute — always prefer it.
- Combine with tensor parallelism inside each stage to exploit fast NVLink bandwidth, leaving the slower inter-node link for pipeline communication.
