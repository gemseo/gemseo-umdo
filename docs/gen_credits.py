# ISC License
#
# Copyright (c) 2021, Timothée Mazzucotelli
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# Adapted from https://github.com/mkdocstrings/griffe/blob/main/scripts/gen_credits.py
"""Script to generate the project's credits."""

from __future__ import annotations

import operator
import os
import re
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import metadata
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

if TYPE_CHECKING:
    from collections.abc import Mapping

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

project_dir = Path(os.getenv("MKDOCS_CONFIG_DIR", "."))
with project_dir.joinpath("pyproject.toml").open("rb") as pyproject_file:
    pyproject = tomllib.load(pyproject_file)
project = pyproject["project"]
project_name = project["name"]
regex = re.compile(r"(?P<dist>[\w.-]+)(?P<spec>.*)$")


def _get_license(pkg_name: str) -> str:
    try:
        data = metadata(pkg_name)
    except PackageNotFoundError:
        # Typically for the test dependencies that are not installed
        # in the doc environment.
        pass
    else:
        license_name = data.get("License", "").strip()

        # Do not use the full license text (with line returns).
        if license_name and "\n" not in license_name and license_name != "UNKNOWN":
            return license_name

        # Look for the classifier.
        for header, value in data.items():
            if header == "Classifier" and value.startswith("License ::"):
                license_name = value.rsplit("::", 1)[1].strip()
                break

        if license_name:
            return license_name

    # Look for the classifier from PyPI.
    response = requests.get(f"https://pypi.org/pypi/{pkg_name}/json").json()
    for value in response["info"]["classifiers"]:
        if value.startswith("License ::"):
            return value.rsplit("::", 1)[1].strip()

    return "?"


def _get_deps(base_deps: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    deps = {}
    for dep in base_deps:
        parsed = regex.match(dep).groupdict()
        dep_name = parsed["dist"].lower()
        deps[dep_name] = {
            "license": _get_license(dep_name),
            "url": f"https://pypi.org/project/{dep_name}/",
        }
    return deps


def _render_credits() -> str:
    external_dependencies = _get_deps(project["dependencies"])
    external_applications = _get_deps(
        chain(
            *project["optional-dependencies"].values(),
            (
                "commitizen",
                "docformatter",
                "pre-commit",
                "ruff",
                "setuptools",
                "setuptools-scm",
            ),
        )
    )

    # Add softwares not in PyPI.
    external_applications.update({
        "insert-license": {
            "url": "https://github.com/Lucas-C/pre-commit-hooks",
            "license": "MIT",
        },
        "pygrep-hooks": {
            "url": "https://github.com/pre-commit/pygrep-hooks",
            "license": "MIT",
        },
    })
    external_dependencies["Python"] = {
        "url": "https://python.org",
        "license": "Python Software License",
    }

    # Add name for later sorting and rendering.
    for deps in (external_dependencies, external_applications):
        for name, dep in deps.items():
            dep["name"] = name

    template_data = {
        "project_name": project_name,
        "external_dependencies": sorted(
            external_dependencies.values(), key=operator.itemgetter("name")
        ),
        "external_applications": sorted(
            external_applications.values(), key=operator.itemgetter("name")
        ),
    }

    jinja_env = SandboxedEnvironment(undefined=StrictUndefined)
    template_text = Path(project_dir / "docs" / "credits.md.jinja").read_text()
    return jinja_env.from_string(template_text).render(**template_data)


print(_render_credits())  # noqa: T201
