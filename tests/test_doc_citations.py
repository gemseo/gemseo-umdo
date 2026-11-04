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

import markdown
import pytest

bib_path = Path(__file__, "..", "..", "docs", "references.bib").resolve()


def render(text: str, extension: str = "footnotes") -> str:
    """Render a Markdown text with the citation extension.

    Args:
        text: The Markdown text.
        extension: The extension used in addition to the citation one.

    Returns:
        The HTML text.
    """
    return markdown.markdown(
        text,
        extensions=[extension, "gemseo_docs.bibtex_citations"],
        extension_configs={"gemseo_docs.bibtex_citations": {"bib_file": str(bib_path)}},
    )


def test_citation() -> None:
    """Check that a citation becomes a footnote with the formatted reference."""
    html = render("A text [@Mura2020].")
    assert 'href="#fn:cite-Mura2020"' in html
    assert "Riccardo Mura" in html
    assert "[@" not in html


def test_citation_block() -> None:
    """Check that a block citing several references with affixes becomes footnotes."""
    html = render("A text [see @giles2015multilevel, p. 1; @Mura2020].")
    assert 'href="#fn:cite-giles2015multilevel"' in html
    assert 'href="#fn:cite-Mura2020"' in html
    assert "see" not in html.split('<div class="footnote">')[0]


def test_citation_in_code_block() -> None:
    """Check that a citation in a fenced code block is left untouched."""
    html = render("```\n[@Mura2020]\n```", "pymdownx.superfences")
    assert "[@Mura2020]" in html


def test_unknown_key(caplog) -> None:
    """Check that a block citing an unknown key is left untouched with a warning."""
    assert "[@unknown]" in render("[@unknown]")
    assert "The citation keys ['unknown'] are not in the bibliography." in caplog.text


@pytest.mark.parametrize(
    "text",
    [
        "Use `[@Mura2020]` here.",
        "Use ``[@Mura2020]`` here.",
        "Ask [@Mura2020](https://example.org).",
        "Ask [@Mura2020][ref].",
        "See [the @Mura2020 docs].",
        "Write to [contact@gemseo.org].",
    ],
    ids=[
        "inline-code",
        "double-backtick",
        "inline-link",
        "reference-link",
        "free-text",
        "email",
    ],
)
def test_not_a_citation(text) -> None:
    """Check that inline code, link text, free text and emails are not citations."""
    html = render(text)
    assert "fn:cite-" not in html
    assert "@" in html


def test_citation_next_to_inline_code() -> None:
    """Check that a citation next to inline code is rendered."""
    html = render("Use `[@Mura2020]` [@Mura2020].")
    assert "<code>[@Mura2020]</code>" in html
    assert 'href="#fn:cite-Mura2020"' in html


def test_footnote_collision() -> None:
    """Check that a footnote of the page named like a cited key is kept."""
    html = render("A note[^Mura2020] and [@Mura2020].\n\n[^Mura2020]: My own note.")
    assert "My own note." in html
    assert "Riccardo Mura" in html
