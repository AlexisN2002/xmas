## What and why

<!-- What changes, and what problem it solves. Link the issue if there is one. -->

## How to verify

<!-- The commands or steps a reviewer runs to see it working. -->

## Checklist

- [ ] Title follows [Conventional Commits](https://www.conventionalcommits.org/) (it becomes the squash commit message)
- [ ] Behaviour changes are covered by a test that would fail without this PR
- [ ] `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy` pass locally
- [ ] Public API is typed and has Google-style docstrings
- [ ] No credentials, run artifacts or datasets added to the repo
