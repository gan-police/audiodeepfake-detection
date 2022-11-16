"""This module implements our CI function calls.
See also: https://nox.thea.codes/en/stable/
"""
import nox

# Evt: Test pipeline aufsetzen. 
#@nox.session(name="test")
#def run_test(session):
#    """Run pytest."""
#    session.install("-r", "requirements.txt")
#    session.install("pytest")
#    session.run("pytest")

# TODO: Requirements.txt anlegen.
@nox.session(name="lint")
def lint(session):
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
def mypy(session):
    """Check type hints."""
    session.install("-r", "requirements.txt")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--no-strict-optional",
        "--no-warn-return-any",
        "--implicit-reexport",
        "--allow-untyped-calls",
        "src",
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "tests", "noxfile.py")
    session.run("black", "src", "tests", "noxfile.py")