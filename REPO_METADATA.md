# Repository Identity Card

This file documents the GitHub-facing metadata for the **hpc-llm-forge** repository. Paste these fields into the GitHub repository settings.

---

## Repository Name

```
hpc-llm-forge
```

**Rationale.** Three syllables, easy to recall, evokes both *forging* (low-level, hand-crafted infrastructure) and the *furnace* in which large-language-model systems are built. The `hpc-` prefix signals the systems-engineering posture; the `-forge` suffix matches the established convention for tooling repositories (e.g., `huggingface/text-generation-inference`).

---

## Short Description (GitHub Profile Tagline)

> **A from-scratch engineering codex for high-performance LLM training — Ring-AllReduce, ZeRO, 1F1B pipelining, and Dockerized multi-cloud launch, derived equation by equation.**

*(Character count: 192 — within GitHub's 350-char limit; reads cleanly on the profile pinned-repository card.)*

---

## Alternative Tagline Variants

For A/B testing on the profile, three additional one-liners:

1. *Distributed training, ZeRO partitioning, and cloud orchestration — implemented from first principles, not from a wrapper.*
2. *A production-grade reference implementation of every parallelism strategy that powers frontier-scale LLMs.*
3. *Where the math meets the metal: hand-derived DDP, tensor parallelism, and ZeRO with empirical validation on H100/B200.*

---

## Topics / Tags

Recommended GitHub topic list (paste into the "Topics" field, comma-separated):

```
llm
llmops
distributed-training
deepspeed
zero-optimization
data-parallelism
tensor-parallelism
pipeline-parallelism
mlops
gpu-computing
cuda
pytorch
mixed-precision
fp8
ring-allreduce
megatron-lm
high-performance-computing
hpc
docker
runpod
lambda-labs
ai-infrastructure
foundation-models
from-scratch
```

---

## Suggested Social Preview Image

Place a 1280 × 640 PNG at `assets/social-preview.png`. Recommended composition:

- Black background with a faint roofline-model plot overlay.
- Centered title: **HPC-LLM-Forge**.
- Sub-title: *Distributed Training · ZeRO · Pipeline Parallelism · Cloud Orchestration*.
- Bottom-right: small repo URL `github.com/HAYDARKILIC/hpc-llm-forge`.

---

## Recommended Repository Settings

| Setting                          | Value                                          |
|----------------------------------|------------------------------------------------|
| Visibility                       | Public                                         |
| Default branch                   | `main`                                         |
| Issues                           | Enabled                                        |
| Discussions                      | Enabled (for theory Q&A)                       |
| Wiki                             | Disabled (theory lives in `docs/theory/`)      |
| Sponsorships                     | Optional                                       |
| Branch protection on `main`      | Require PR + passing CI + ≥ 1 review           |
| Required CI checks               | `lint`, `test`, `type-check`                   |
| Releases tagging                 | Semantic versioning (`v0.1.0`, `v0.2.0`, …)    |
