# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Tests of the Python-Markdown extension rendering the citations as footnotes."""

from __future__ import annotations

from pathlib import Path

import pytest
from gemseo.util.testing.helper import assert_exception

markdown = pytest.importorskip("markdown")
pytest.importorskip("gemseo_docs.bibtex_citations")

bib_path = Path(__file__, "..", "..", "docs", "references.bib").resolve()


def render(text: str) -> str:
    """Render a Markdown text with the citation extension.

    Args:
        text: The Markdown text.

    Returns:
        The HTML text.
    """
    return markdown.markdown(
        text,
        extensions=["footnotes", "gemseo_docs.bibtex_citations"],
        extension_configs={"gemseo_docs.bibtex_citations": {"bib_file": str(bib_path)}},
    )


def test_citation() -> None:
    """Check that a citation becomes a footnote with the formatted reference."""
    html = render("A text [@Mura2020].")
    assert 'href="#fn:Mura2020"' in html
    assert "Riccardo Mura" in html
    assert "[@" not in html


def test_citation_block() -> None:
    """Check that a block citing several references with affixes becomes footnotes."""
    html = render("A text [see @giles2015multilevel, p. 1; @Mura2020].")
    assert 'href="#fn:giles2015multilevel"' in html
    assert 'href="#fn:Mura2020"' in html
    assert "see" not in html.split('<div class="footnote">')[0]


def test_email_is_not_a_citation() -> None:
    """Check that a bracketed email address is left untouched."""
    assert "[contact@gemseo.org]" in render("Write to [contact@gemseo.org].")


def test_citation_in_code_block() -> None:
    """Check that a citation in a fenced code block is left untouched."""
    html = markdown.markdown(
        "```\n[@Mura2020]\n```",
        extensions=["pymdownx.superfences", "gemseo_docs.bibtex_citations"],
        extension_configs={"gemseo_docs.bibtex_citations": {"bib_file": str(bib_path)}},
    )
    assert "[@Mura2020]" in html


def test_unknown_key(snapshot) -> None:
    """Check that citing an unknown key raises an error."""
    with assert_exception(KeyError, snapshot):
        render("[@unknown]")
