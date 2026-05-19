# High-Performance AI Infrastructure & LLMOps

A from-scratch engineering codex for the **High-Performance AI Infrastructure & LLMOps** course — a six-week trajectory through the systems that train frontier-scale LLMs.

Every algorithm — Ring-AllReduce, tensor and pipeline parallelism, ZeRO Stage 1/2/3, CPU/NVMe offload — is derived from first principles and implemented in raw PyTorch before any production framework is invoked.

## Required Reading

- **Kirk, D. B. & Hwu, W.-M. W.** — *Programming Massively Parallel Processors* (4th ed., Morgan Kaufmann). Foundation for Weeks 1–3.
- **Raj, E.** — *Engineering MLOps* (Packt Publishing). Foundation for Weeks 5–6.

## Curriculum

| Week | Topic | Notebook |
|------|-------|----------|
| 1 | GPU memory hierarchy, Tensor Cores, precision lattice, the Roofline model | `01_gpu_memory_hierarchy.ipynb` |
| 2 | Data parallelism — `DataParallel` vs. DDP, Ring-AllReduce, gradient bucketing | `02_ddp_from_scratch.ipynb` |
| 3 | Tensor parallelism (Megatron-LM) and 1F1B pipeline scheduling | `03_tensor_pipeline_parallel.ipynb` |
| 4 | DeepSpeed and ZeRO Stage 1/2/3 with CPU/NVMe offload | `04_deepspeed_zero.ipynb` |
| 5 | Dockerized cloud orchestration on RunPod / Lambda / Vast.ai | `05_cloud_orchestration.ipynb` |
| 6 | Capstone — end-to-end multi-node training pipeline | `06_capstone_pipeline.ipynb` |

Each notebook has a paired derivation in `docs/theory/`.

## Layout

```
src/hpc_ai_infra_llmops/    Implementation package
notebooks/          Six course notebooks
docs/theory/        Math derivations
configs/            DeepSpeed JSON + training YAML
docker/             CUDA 12.4 / PyTorch 2.4 image
scripts/            DDP / DeepSpeed / cloud launchers
benchmarks/         AllReduce, ZeRO memory, precision sweeps
examples/           Capstone training script
```

## Setup

Requires an NVIDIA GPU (compute capability ≥ 7.0), CUDA 12.4+, Python 3.11+.

```bash
git clone https://github.com/HAYDARKILIC/hpc_ai_infra_llmops.git
cd hpc_ai_infra_llmops
pip install -e ".[dev]"
make test
```

Docker:

```bash
make docker-build
make docker-shell
```

## Usage

```bash
# Week 2 — DDP on 4 GPUs
torchrun --standalone --nproc_per_node=4 \
    examples/train_gpt_capstone.py \
    --config configs/training/gpt_350m.yaml --parallelism ddp

# Week 4 — ZeRO-3 + CPU offload on 8 GPUs
deepspeed --num_gpus=8 \
    examples/train_gpt_capstone.py \
    --config configs/training/gpt_1b3.yaml --parallelism zero \
    --deepspeed configs/deepspeed/zero3_offload.json

# Week 6 — one-command cloud launch
export RUNPOD_API_KEY=... WANDB_API_KEY=...
make launch-cloud PROVIDER=runpod GPU=H100 NUM_GPUS=8
```

## Author

Haydar Kılıç — [github.com/HAYDARKILIC](https://github.com/HAYDARKILIC)
