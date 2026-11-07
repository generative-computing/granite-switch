.PHONY: test test-unit test-composer test-hf test-vllm test-integration test-all test-cpu test-gpu test-gpu-full test-tp test-e2e test-regression-fast test-regression-real test-regression-hf test-peft-equiv test-peft-equiv-hf lint fmt pre-commit help

# Default Python (can override: make test PYTHON=python3.11)
PYTHON ?= python

# Pytest flags per CLAUDE.md guidelines
PYTEST_FLAGS = -v -s --tb=short

# ── The real-checkpoint (e2e) gate ───────────────────────────────────────────
# Eleven modules across tests/hf, tests/integration and tests/vllm compose a real
# ~3B Granite plus adapters, and skip themselves unless GRANITE_SWITCH_E2E_MODELS=1.
# That flag is an opt-in for COST, not a capability check: nothing is wrong with the
# machine when they skip, they are just expensive (a multi-GB download on a cold
# cache). `make test-e2e` is the supported way to turn them on, so the knowledge of
# which variable that is lives here instead of in everyone's shell history.
#
# GRANITE_SWITCH_E2E_DIR is where composed checkpoints are cached between runs; the
# modules warm-reuse any dir that already has a config.json, so a second `make
# test-e2e` pays the compose cost once. Override it to keep the cache off /tmp:
#     make test-e2e GRANITE_SWITCH_E2E_DIR=/scratch/gs-e2e
#
# `?=` on both, so an already-exported value wins and CI can set them in the
# environment instead.
GRANITE_SWITCH_E2E_MODELS ?= 1
GRANITE_SWITCH_E2E_DIR ?= /tmp/granite_switch_e2e
E2E_ENV = GRANITE_SWITCH_E2E_MODELS=$(GRANITE_SWITCH_E2E_MODELS) \
          GRANITE_SWITCH_E2E_DIR=$(GRANITE_SWITCH_E2E_DIR)

# Individual test suites
test-unit:
	$(PYTHON) -m pytest tests/unit/ $(PYTEST_FLAGS)

test-composer:
	$(PYTHON) -m pytest tests/composer/ $(PYTEST_FLAGS)

test-hf:
	$(PYTHON) -m pytest tests/hf/ $(PYTEST_FLAGS)

test-vllm:
	$(PYTHON) -m pytest tests/vllm/ $(PYTEST_FLAGS)

test-integration:
	$(PYTHON) -m pytest tests/integration/ $(PYTEST_FLAGS)

# Combined targets
test: test-unit test-composer  # CPU tests only (default)

test-cpu: test-unit test-composer test-hf  # All CPU tests

test-gpu: test-vllm test-integration  # GPU-required tests

test-gpu-full: test-gpu test-regression-fast  # GPU tests + regression suite

test-tp:
	$(PYTHON) -m pytest tests/vllm/test_tp_integration.py tests/vllm/test_tp_lora.py $(PYTEST_FLAGS)

# Everything the integration suite covers, WITH the real-checkpoint modules enabled.
#
# Three directories, not just tests/integration/: the e2e-gated modules are spread
# across tests/hf (4), tests/integration (5) and tests/vllm (1), so running only
# tests/integration/ would leave six of them still skipping and the target would
# quietly under-deliver.
#
# Separate pytest invocations on purpose. These modules each build a vLLM engine or
# serve a model, and a per-directory process keeps one leaking CUDA context from
# poisoning the next. No leading `-`: unlike the regression targets below, a failure
# here should stop the run — it means a real checkpoint behaved wrongly.
test-e2e:
	$(E2E_ENV) $(PYTHON) -m pytest tests/hf/ $(PYTEST_FLAGS)
	$(E2E_ENV) $(PYTHON) -m pytest tests/vllm/ $(PYTEST_FLAGS)
	$(E2E_ENV) $(PYTHON) -m pytest tests/integration/ $(PYTEST_FLAGS)

test-all: test-cpu test-gpu  # Everything

# Regression test suites (no -x: run all tests even if some fail)
test-regression-fast:
	-$(PYTHON) -m pytest tests/regression/hf/test_generation_regression.py $(PYTEST_FLAGS)
	-$(PYTHON) -m pytest tests/regression/vllm/test_generation_regression.py $(PYTEST_FLAGS)
	-$(PYTHON) -m pytest tests/regression/integration/test_cross_backend_regression.py $(PYTEST_FLAGS)

test-regression-real:
	-$(PYTHON) -m pytest tests/regression/hf/test_generation_regression_real.py $(PYTEST_FLAGS)
	-$(PYTHON) -m pytest tests/regression/vllm/test_generation_regression_real.py $(PYTEST_FLAGS)

test-peft-equiv:
	-$(PYTHON) -m pytest tests/regression/hf/test_peft_equivalence.py $(PYTEST_FLAGS)
	-$(PYTHON) -m pytest tests/regression/vllm/test_peft_equivalence.py $(PYTEST_FLAGS)

# HF-only regression (no vLLM/GPU needed)
test-regression-hf:
	-$(PYTHON) -m pytest tests/regression/hf/test_generation_regression.py $(PYTEST_FLAGS)
	-$(PYTHON) -m pytest tests/regression/hf/test_generation_regression_real.py $(PYTEST_FLAGS)
	-$(PYTHON) -m pytest tests/regression/hf/test_peft_equivalence.py $(PYTEST_FLAGS)

test-peft-equiv-hf:
	-$(PYTHON) -m pytest tests/regression/hf/test_peft_equivalence.py $(PYTEST_FLAGS)

# Linting / formatting
lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff format .

pre-commit:
	uv run pre-commit run --all-files

# Help
help:
	@echo "Available targets:"
	@echo "  test           - Run unit + composer tests (CPU, fast)"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-composer  - Run composer tests only"
	@echo "  test-hf        - Run HuggingFace tests (CPU)"
	@echo "  test-vllm      - Run vLLM tests (requires GPU)"
	@echo "  test-integration - Run integration tests (requires GPU)"
	@echo "  test-cpu       - Run all CPU tests"
	@echo "  test-gpu       - Run all GPU tests"
	@echo "  test-gpu-full  - Run GPU tests + regression-fast"
	@echo "  test-tp        - Run TP integration + TP LoRA tests (requires 2+ GPUs)"
	@echo "  test-e2e       - Run hf + vllm + integration WITH the real-checkpoint"
	@echo "                   (GRANITE_SWITCH_E2E_MODELS) modules enabled. Composes a"
	@echo "                   real ~3B checkpoint; cached in GRANITE_SWITCH_E2E_DIR"
	@echo "                   (default /tmp/granite_switch_e2e). Requires GPU."
	@echo "  test-all       - Run all tests"
	@echo "  test-regression-fast  - Run fast synthetic regression tests"
	@echo "  test-regression-real  - Run real model regression tests (requires_model)"
	@echo "  test-peft-equiv       - Run PEFT equivalence tests (HF + vLLM)"
	@echo "  test-regression-hf    - Run all HF regression tests (no vLLM/GPU)"
	@echo "  test-peft-equiv-hf    - Run PEFT equivalence (HF only)"
	@echo "  lint           - Run ruff linter"
	@echo "  fmt            - Run ruff formatter"
	@echo "  pre-commit     - Run all pre-commit hooks on all files"
