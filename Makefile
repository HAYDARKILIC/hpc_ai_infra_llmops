# ---------------------------------------------------------------------------
# HPC-LLM-Forge — developer Makefile
# ---------------------------------------------------------------------------

PYTHON         ?= python
PIP            ?= $(PYTHON) -m pip
DOCKER_IMAGE   ?= hpc-llm-forge:latest
NPROC          ?= 1
CONFIG         ?= configs/training/gpt_350m.yaml

.PHONY: help install install-dev lint format test bench docker-build docker-shell \
        docker-test launch-ddp launch-deepspeed launch-cloud clean

help:
	@echo "HPC-LLM-Forge — available targets"
	@echo "  install         Install runtime dependencies"
	@echo "  install-dev     Install dev + runtime dependencies (editable mode)"
	@echo "  lint            Run ruff, black --check, and mypy"
	@echo "  format          Auto-format with ruff & black"
	@echo "  test            Run the pytest suite with coverage"
	@echo "  bench           Run regression benchmarks"
	@echo "  docker-build    Build the local CUDA image"
	@echo "  docker-shell    Drop into an interactive container shell"
	@echo "  docker-test     Run the test suite inside the container"
	@echo "  launch-ddp      Launch from-scratch DDP training"
	@echo "  launch-deepspeed  Launch DeepSpeed ZeRO training"
	@echo "  launch-cloud    Launch the capstone pipeline on a cloud provider"
	@echo "  clean           Remove caches and build artifacts"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -e ".[dev,deepspeed,cloud]"

lint:
	ruff check src tests
	black --check src tests
	mypy src

format:
	ruff check --fix src tests
	black src tests

test:
	pytest -n auto

bench:
	$(PYTHON) benchmarks/allreduce_throughput.py
	$(PYTHON) benchmarks/zero_memory_sweep.py
	$(PYTHON) benchmarks/precision_pareto.py

docker-build:
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE) .

docker-shell:
	docker run --rm -it --gpus all -v $(PWD):/workspace $(DOCKER_IMAGE) bash

docker-test:
	docker run --rm --gpus all -v $(PWD):/workspace $(DOCKER_IMAGE) make test

launch-ddp:
	torchrun --standalone --nproc_per_node=$(NPROC) \
		examples/train_gpt_capstone.py --config $(CONFIG) --parallelism ddp

launch-deepspeed:
	deepspeed --num_gpus=$(NPROC) examples/train_gpt_capstone.py \
		--config $(CONFIG) --deepspeed configs/deepspeed/zero3_offload.json

launch-cloud:
	bash scripts/launch_cloud.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info \
	       **/__pycache__ .coverage coverage.xml
