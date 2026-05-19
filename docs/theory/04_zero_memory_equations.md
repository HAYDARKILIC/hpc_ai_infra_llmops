# ZeRO Memory Equations

## 1. Mixed-precision Adam baseline

For an $N$-parameter model trained with mixed-precision Adam, every parameter requires storage in two forms:

| Quantity                | Precision | Bytes |
|-------------------------|-----------|-------|
| Working parameter       | FP16/BF16 | 2     |
| Working gradient        | FP16/BF16 | 2     |
| Master parameter        | FP32      | 4     |
| Adam first moment $m$   | FP32      | 4     |
| Adam second moment $v$  | FP32      | 4     |

Total per-parameter state: **16 bytes**. Without partitioning, every GPU pays this cost in full:

$$
M_{\text{baseline}} \;=\; 16 N + M_{\text{activations}}.
$$

## 2. Stage 1 — partition optimizer state

Stage 1 distributes the 12 fp32 bytes (master + $m$ + $v$) across $N_{\text{GPU}}$ ranks:

$$
M_{\text{Stage1}} \;=\; 2N + 2N + \frac{12 N}{N_{\text{GPU}}} + M_{\text{act}}.
$$

Communication overhead vs. baseline: one extra **all-gather** of the updated parameters per step.

## 3. Stage 2 — also partition gradients

Stage 2 replaces the all-reduce on gradients with a **reduce-scatter**: each rank receives only the slice of the gradient it owns:

$$
M_{\text{Stage2}} \;=\; 2N + \frac{2 N + 12 N}{N_{\text{GPU}}} + M_{\text{act}}.
$$

Communication overhead vs. baseline: the all-reduce is split into reduce-scatter + all-gather, but the total bytes transferred are unchanged. No additional cost.

## 4. Stage 3 — also partition parameters

Stage 3 partitions the fp16 parameters too. During the forward pass each layer's parameters are **all-gathered** just in time, used for that layer's matmul, and released immediately:

$$
M_{\text{Stage3}} \;=\; \frac{16 N}{N_{\text{GPU}}} + M_{\text{act}}.
$$

Communication overhead: **one extra all-gather per layer per forward pass** (and again per backward pass). For an $L$-layer model the extra bytes per step are $\sim 2L \cdot (2 N / N_{\text{GPU}})$ — i.e. each parameter is moved $\sim 4L / N_{\text{GPU}}$ times instead of $\sim 2$ times in vanilla DDP.

## 5. Numerical example: $N = 1.3 \times 10^9$, $N_{\text{GPU}} = 8$

| Stage | param (GiB) | grad (GiB) | opt (GiB) | total (GiB) |
|-------|-------------|------------|-----------|-------------|
| 0     | 2.42        | 2.42       | 14.55     | **19.39**   |
| 1     | 2.42        | 2.42       | 1.82      | 6.66        |
| 2     | 2.42        | 0.30       | 1.82      | 4.54        |
| 3     | 0.30        | 0.30       | 1.82      | **2.42**    |

Stage 3 reduces state memory by **8×** (matching $1/N_{\text{GPU}}$).

## 6. Activation memory

The above counts only **persistent state**. Activations follow a separate $O(L \cdot B \cdot s \cdot d)$ scaling and are addressed orthogonally by:

- **Activation checkpointing.** Memory $O(\sqrt{L})$ at $\sim 33\%$ compute overhead.
- **Sequence parallelism.** Partitions the activation tensors along the sequence dimension, dividing activation memory by the tensor-parallel size.
- **Selective checkpointing.** Re-compute only the cheapest activations (e.g. element-wise activations) and persist the expensive ones (e.g. attention scores).

## 7. CPU / NVMe offload

ZeRO-Infinity extends Stage 3 by **offloading** the partitioned state to host memory or NVMe SSD. The feasibility condition is

$$
B_{\text{offload}} \cdot t_{\text{step}} \;\gtrsim\; M_{\text{offloaded}}.
$$

With CPU offload over PCIe Gen5 x16 ($B \approx 64$ GB/s) and a $t_{\text{step}}$ of $\sim 1\text{ s}$, up to $\sim 64$ GiB of state can be offloaded per step without slowing the GPU — enough to fit a 30B-parameter model on a single A100-80GB.
