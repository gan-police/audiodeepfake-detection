"""This module implements our CI function calls.

Run with 'nox -r -v -s name'.
See also: https://nox.thea.codes/en/stable/
"""
import nox


@nox.session(name="test")
def run_test(session) -> None:
    """Run tests."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest", "tests")


@nox.session(name="lint")
def lint(session) -> None:
    """Check code conventions."""
    session.install("flake8==4.0.1")
    session.install(
        "flake8-colors",
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.install("flake8-bandit==2.1.2", "bandit==1.7.2")
    session.run("flake8", "src", "scripts", "noxfile.py")


@nox.session(name="typing")
def mypy(session) -> None:
    """Check type hints."""
    session.install("-r", "requirements.txt")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        # "--no-strict-optional",
        # "--no-warn-return-any",
        "--implicit-reexport",
        "--allow-untyped-calls",
        "src",
    )


@nox.session(name="format")
def format(session) -> None:
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "scripts", "noxfile.py")
    session.run("black", "src", "scripts", "noxfile.py")
