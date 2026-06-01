# Contributing to Granite Switch

Thank you for your interest in contributing to Granite Switch!

## Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it once before working on the project:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via pip: `pip install uv`

## Getting Started

1. Fork the repository
2. Clone your fork and install dependencies:
   ```bash
   git clone https://github.com/<your-username>/granite-switch.git
   cd granite-switch
   uv sync --group dev
   ```
3. Create a feature branch and make your changes
4. Run tests: `uv run pytest tests/ -v`
5. Submit a pull request

## Contribution Guidelines

See [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for detailed workflow, commit conventions, and code quality standards.

## Areas of Interest

- **Bug fixes** — Identify and fix issues in the codebase
- **Documentation** — Improve tutorials and guides

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

## Questions?

Open an issue or start a discussion.
