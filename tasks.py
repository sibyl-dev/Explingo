import subprocess
import webbrowser
import shutil

from pathlib import Path
from invoke import task
from sys import executable
import os


def print_red(s):
    print("\033[91m {}\033[00m".format(s), end="")


def print_green(s):
    print("\033[92m {}\033[00m".format(s), end="")


@task
def clean_test(context):
    """
    Cleans the test store
    """

    shutil.rmtree(Path(".pytest_cache"), ignore_errors=True)


@task
def fix_lint(context):
    """
    Fixes all linting and import sort errors. Skips init.py files for import sorts
    """

    subprocess.run(["black", "explingo"])
    subprocess.run(["black", "tests"])
    subprocess.run(["isort", "--atomic", "explingo", "tests"])


@task
def lint(context):
    """
    Runs the linting and import sort process on all library files and tests and prints errors.
        Skips init.py files for import sorts
    """
    subprocess.run(["flake8", "explingo", "tests"], check=True)
    subprocess.run(["isort", "explingo", "tests"], check=True)


@task
def test(context):
    """
    Runs all test commands.
    """

    failures_in = []

    try:
        test_unit(context)
    except subprocess.CalledProcessError:
        failures_in.append("Unit tests")

    if len(failures_in) == 0:
        print_green("\nAll tests successful :)")
    else:
        print_red("\n:( Failures in: ")
        for i in failures_in:
            print_red(i + ", ")


@task
def test_unit(context):
    """
    Runs all unit tests and outputs results and coverage
    """
    subprocess.run(["pytest"], check=True)
