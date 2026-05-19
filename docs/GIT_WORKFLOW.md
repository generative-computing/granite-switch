# Contributing Guide

Guidelines for contributing to Granite Switch.

## Quick Start

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes and commit
4. Push and open a Pull Request

## Branching

- **`main`**: Stable branch, always ready for release
- **Feature branches**: `feature/short-description`
- **Bugfix branches**: `bugfix/short-description`

## Workflow

```bash
# 1. Create branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature

# 2. Make changes and commit
git add <files>
git commit -m "Add feature X"

# 3. Keep up-to-date with main
git fetch origin
git rebase origin/main

# 4. Push and create PR
git push origin feature/your-feature
```

## Commit Messages

Write clear commit messages that explain **what** changed and **why**:

```
Short summary (50 chars or less)

Longer explanation if needed. Explain what changed and why,
not how (the diff shows how).

Fixes #123
```

**Good examples:**
- "Fix batch indexing for variable sequence lengths"
- "Add serialization roundtrip test"
- "Update supported models documentation"

**Avoid:**
- "fix bug" (what bug?)
- "update code" (what changed?)
- "WIP" (squash before merging)

## Code Quality

Before committing:

1. **Run tests**: `uv run pytest tests/ -v`
2. **Check comments match code** — stale comments are worse than no comments
3. **Update docs** if behavior changed

## Pull Requests

- Target the `main` branch
- Include a clear description of changes
- Reference related issues
- Ensure tests pass

## Questions?

Open an issue or start a discussion.
