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
"""A Python-Markdown extension rendering the BibTeX citations as footnotes.

It replaces the mkdocs-bibtex plugin, which Zensical does not support,
and reproduces its simple mode:
a citation block ``[@key]`` (or ``[@key1; @key2]``)
becomes the footnote reference ``[^key]``
and the page ends with the footnotes ``[^key]: <formatted reference>``,
formatted with the plain style of pybtex.
The prefixes and the suffixes of the citations are ignored.
The parts of a block including an email address are ignored,
so ``[contact@gemseo.org]`` is not a citation.

The citation blocks in fenced code blocks are left untouched,
as these blocks are stashed by ``pymdownx.superfences`` beforehand.
"""

from __future__ import annotations

import re
from functools import cache
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from markdown import Extension
from markdown.preprocessors import Preprocessor
from pybtex.backends.markdown import Backend as MarkdownBackend
from pybtex.database import parse_file
from pybtex.style.formatting.plain import Style as PlainStyle

if TYPE_CHECKING:
    from markdown import Markdown
    from pybtex.database import BibliographyData

citation_block_regex = re.compile(r"\[(.*?)\]")
"""The regular expression to find a citation block, e.g. ``[@foo; @bar]``."""

citation_regex = re.compile(
    r"(?:(?P<prefix>[^@;]*?)\s*)?@(?P<key>[\w-]+)(?:,\s*(?P<suffix>[^;]+))?"
)
"""The regular expression that a citation must match, e.g. ``see @foo, p. 1``.

As in the simple mode of mkdocs-bibtex, the prefix and the suffix are not rendered.
"""

email_regex = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
"""The regular expression that an email address matches."""


@cache
def read_bibliography(path: str) -> BibliographyData:
    """Read a BibTeX file once, as an extension is created for each Markdown instance.

    Args:
        path: The path to the BibTeX file.

    Returns:
        The bibliography.
    """
    return parse_file(path)


class CitationPreprocessor(Preprocessor):
    """A preprocessor replacing the citation blocks by footnotes."""

    def __init__(self, md: Markdown, bibliography: BibliographyData) -> None:
        """
        Args:
            md: The Markdown instance.
            bibliography: The bibliography.
        """  # noqa: D205, D212
        super().__init__(md)
        self.__bibliography = bibliography
        self.__style = PlainStyle()
        self.__backend = MarkdownBackend()

    def run(self, lines: list[str]) -> list[str]:
        """Replace the citation blocks by footnotes.

        Args:
            lines: The lines of the page.

        Returns:
            The lines of the page with footnotes instead of citation blocks.

        Raises:
            KeyError: When a citation refers to an unknown key.
        """
        cited_keys = {}

        def replace(match: re.Match[str]) -> str:
            """Replace a citation block by footnote references.

            Args:
                match: The match of a candidate citation block.

            Returns:
                The footnote references,
                or the original text when the block is not a citation block.
            """
            parts = [
                part
                for part in match.group(1).split(";")
                if email_regex.search(part) is None
            ]
            matches = [citation_regex.match(part) for part in parts]
            if not parts or None in matches:
                return match.group(0)

            keys = [citation.group("key") for citation in matches]
            for key in keys:
                if key not in self.__bibliography.entries:
                    msg = f"The citation key '{key}' is not in the bibliography."
                    raise KeyError(msg)

                cited_keys[key] = None

            return "".join(f"[^{key}]" for key in keys)

        lines = [citation_block_regex.sub(replace, line) for line in lines]
        if cited_keys:
            lines.append("")
            lines.extend(f"[^{key}]: {self.__format(key)}" for key in cited_keys)

        return lines

    def __format(self, key: str) -> str:
        """Format a reference.

        Args:
            key: The key of the reference.

        Returns:
            The formatted reference.
        """
        entry = self.__bibliography.entries[key]
        text = self.__style.format_entry("", entry).text.render(self.__backend)
        return (
            text
            .replace("\n", " ")
            .replace("\\(", "(")
            .replace("\\)", ")")
            .replace("\\.", ".")
        )


class CitationExtension(Extension):
    """A Python-Markdown extension rendering the BibTeX citations as footnotes."""

    priority: ClassVar[int] = 20
    """The priority of the preprocessor.

    It must be lower than the one of ``pymdownx.superfences`` (25),
    so that the fenced code blocks are already stashed,
    and than the one of ``pymdownx.snippets`` (32),
    so that the included files are processed too.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Args:
            **kwargs: The configuration of the extension.
        """  # noqa: D205, D212
        self.config = {"bib_file": ["", "The path to the BibTeX file."]}
        super().__init__(**kwargs)

    def extendMarkdown(self, md: Markdown) -> None:  # noqa: N802
        """Register the preprocessor.

        Args:
            md: The Markdown instance.
        """
        bibliography = read_bibliography(self.getConfig("bib_file"))
        md.preprocessors.register(
            CitationPreprocessor(md, bibliography), "bibtex_citations", self.priority
        )


def makeExtension(**kwargs: Any) -> CitationExtension:  # noqa: N802
    """Create the extension.

    Args:
        **kwargs: The configuration of the extension.

    Returns:
        The extension.
    """
    return CitationExtension(**kwargs)
