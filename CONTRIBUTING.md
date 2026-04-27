# Contributing

Thank you for helping improve Tensor Network Editor. The project is meant to be
friendly to scientific Python users, so small, clear contributions are very
welcome.

## Development Setup

Use a virtual environment. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install optional extras only when you need them, for example:

```bash
python -m pip install -e ".[dev,planner,tensornetwork,quimb]"
```

## Before Sending Changes

Run these from the repository root:

```bash
python -m ruff check . --fix
python -m ruff format .
python -m mypy
python -m pyright
python -m pytest -q
```

If a command is too slow on your machine, mention which command you ran and what
you were able to verify.

## Coding Guidelines

- Keep changes focused and easy to review.
- Add Python type annotations to every function definition.
- Prefer clear names and explicit data structures over clever shortcuts.
- Keep Windows and Linux behavior in mind when working with paths, subprocesses,
  or shell commands.
- Add or update tests when changing behavior.
- Update `CHANGELOG.md` when the change is user-visible, contributor-visible, or
  fixes a bug that users could hit.
- Avoid adding new Markdown files unless they serve a clear long-term purpose,
  such as project docs, release notes, or contributor guidance.

## Reporting Bugs

Please include:

- What you expected to happen.
- What happened instead.
- The command you ran.
- Your Python version and operating system.
- Any relevant saved design, generated code, or traceback.
