# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""Configuration of mkdocs-gallery, loaded by the docs-tools prebuild."""

from __future__ import annotations

from typing import Any
from typing import Final

from graphviz import Digraph


def _add_graphviz_html_representation() -> None:
    """Display the graphviz graphs of the examples as SVG images.

    mkdocs-gallery displays the last expression of a code block
    with its `_repr_html_()` method or else as text,
    whereas graphviz only provides the IPython mimebundle used by Jupyter.
    """
    Digraph._repr_html_ = lambda self: self.pipe(format="svg", encoding="utf-8")


_add_graphviz_html_representation()

conf: Final[dict[str, Any]] = {}
"""The configuration of mkdocs-gallery completing the one of docs-tools."""
