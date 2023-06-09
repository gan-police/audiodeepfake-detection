"""This module implements our CI function calls.

Run with 'nox -r -v -s name'.
See also: https://nox.thea.codes/en/stable/
"""
import nox


@nox.session(name="format", python=["3.10"])
def format(session) -> None:
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "scripts", "noxfile.py")
    session.run("black", "src", "scripts", "noxfile.py")


@nox.session(name="test", python=["3.10"])
def run_test(session) -> None:
    """Run tests."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest", "tests", external=True)


@nox.session(name="lint", python=["3.10"])
def lint(session) -> None:
    """Check code conventions."""
    session.install("flake8==6.0.0")
    session.install(
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "flake8-bandit",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.run("flake8", "src", "scripts", "noxfile.py")


@nox.session(name="typing", python=["3.10"])
def mypy(session) -> None:
    """Check type hints."""
    session.install("-r", "requirements.txt")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--implicit-reexport",
        "--allow-untyped-calls",
        "src",
    )
