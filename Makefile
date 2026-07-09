.PHONY: test test-unit test-composer test-hf test-vllm test-integration test-all test-cpu test-gpu test-gpu-full test-tp test-regression-fast test-regression-real test-regression-hf test-peft-equiv test-peft-equiv-hf lint fmt pre-commit help

# Default Python (can override: make test PYTHON=python3.11)
PYTHON ?= python

# Pytest flags per CLAUDE.md guidelines
PYTEST_FLAGS = -v -s --tb=short

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
	@echo "  test-all       - Run all tests"
	@echo "  test-regression-fast  - Run fast synthetic regression tests"
	@echo "  test-regression-real  - Run real model regression tests (requires_model)"
	@echo "  test-peft-equiv       - Run PEFT equivalence tests (HF + vLLM)"
	@echo "  test-regression-hf    - Run all HF regression tests (no vLLM/GPU)"
	@echo "  test-peft-equiv-hf    - Run PEFT equivalence (HF only)"
	@echo "  lint           - Run ruff linter"
	@echo "  fmt            - Run ruff formatter"
	@echo "  pre-commit     - Run all pre-commit hooks on all files"
