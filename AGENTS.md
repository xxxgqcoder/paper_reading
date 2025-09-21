# Agent Instructions

This file provides instructions for AI agents working in this repository.

## Build, Lint, and Test

- **Build:** No explicit build step. This is a Python script-based project.
- **Lint:** Run `ruff check .` to lint the codebase.
- **Format:** Run `black .` or `ruff format .` to format the code.
- **Test:** Run `pytest` to run all tests. To run a single test file, use `pytest <path_to_test_file>`.

## Code Style

- **Formatting:** Follow `black` formatting with a line length of 120 characters.
- **Imports:** Imports are sorted using `isort` conventions (via `ruff`).
- **Types:** Type hints are encouraged but not enforced by a static type checker.
- **Naming:** Use `snake_case` for variables and functions, and `PascalCase` for classes.
- **Error Handling:** Handle exceptions explicitly. Avoid broad `except:` clauses.
- **Dependencies:** Manage dependencies in `requirements.txt`.
