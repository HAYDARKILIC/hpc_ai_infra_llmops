"""ZeRO (Zero Redundancy Optimizer) — Stage 1 / 2 / 3 partitioning and offload.

Memory accounting under mixed-precision Adam training of an ``N``-parameter model:

* Parameters (fp16)      : 2N bytes
* Gradients (fp16)       : 2N bytes
* Optimizer state (fp32) : 12N bytes  (master weights 4N + Adam m 4N + Adam v 4N)

Total baseline:           16N bytes per GPU.

ZeRO partitions these into ``N_GPU`` shards:

* **Stage 1.** Partitions optimizer state. Memory per GPU: 4N + 12N / N_GPU.
* **Stage 2.** Stage 1 + partitions gradients. Memory per GPU: 2N + 14N / N_GPU.
* **Stage 3.** Stage 2 + partitions parameters. Memory per GPU: 16N / N_GPU + activations.

See ``docs/theory/04_zero_memory_equations.md`` for the full derivation.
"""
