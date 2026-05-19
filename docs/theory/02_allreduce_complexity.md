# Ring-AllReduce — Bandwidth-Optimality Proof

## 1. Problem statement

$N$ workers each hold a tensor of size $S$ bytes. We require every worker to end with the element-wise sum of all $N$ tensors. Let $C(N, S)$ denote the bytes per worker that any algorithm must transfer in the worst case.

## 2. Lower bound

In an AllReduce, each worker must (i) send its own data to the rest of the system and (ii) receive the contributions of the other $N-1$ workers in some compressed form (otherwise the recipient does not have enough information to compute the sum). A short adversarial argument gives

$$
C(N, S) \;\ge\; 2 S \cdot \frac{N - 1}{N}.
$$

## 3. Achieving the lower bound: Ring-AllReduce

Arrange the $N$ workers in a ring. Split each tensor into $N$ equal chunks of $S/N$ bytes.

### Phase 1 — Reduce-Scatter ($N - 1$ rounds)

In round $t \in \{0, \dots, N-2\}$, worker $r$:

- sends chunk indexed $(r - t) \bmod N$ to its right neighbour,
- receives chunk indexed $(r - t - 1) \bmod N$ from its left neighbour,
- adds the received chunk to its local copy of that chunk.

After $N - 1$ rounds worker $r$ holds the fully-reduced value of chunk $(r + 1) \bmod N$.

Bytes per worker: $(N - 1) \cdot (S / N)$.

### Phase 2 — All-Gather ($N - 1$ rounds)

In round $t$, worker $r$:

- sends its fully-reduced chunk to its right neighbour,
- receives the upstream chunk from its left neighbour.

After $N - 1$ rounds every worker holds every fully-reduced chunk.

Bytes per worker: $(N - 1) \cdot (S / N)$.

### Total

$$
C_{\text{ring}}(N, S) \;=\; 2 (N - 1) \cdot \frac{S}{N} \;=\; 2 S \cdot \frac{N - 1}{N}.
$$

This matches the lower bound, so Ring-AllReduce is bandwidth-optimal.

## 4. Latency cost

Each phase is sequentialised over $N - 1$ rounds, so the latency term is $\alpha \cdot 2(N - 1)$ where $\alpha$ is the per-message latency. For small $S$, tree-based AllReduce wins; for large $S$, the ring wins. Production NCCL switches automatically.

## 5. Practical implication

The per-worker byte count is **independent of $N$** for large $S$. This is why DDP training scales nearly linearly with the GPU count up to the point where compute time falls below the AllReduce time — and that scaling break is what gradient bucketing and overlap are designed to push as far out as possible.
