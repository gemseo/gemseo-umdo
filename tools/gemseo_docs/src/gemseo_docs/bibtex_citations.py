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
becomes the footnote reference ``[^cite-key]``
and the page ends with the footnotes ``[^cite-key]: <formatted reference>``,
formatted with the plain style of pybtex.
The ``cite-`` prefix avoids collisions with the footnotes of the page.
The prefixes and the suffixes of the citations are ignored.
The parts of a block including an email address are ignored,
so ``[contact@gemseo.org]`` is not a citation.

A block is left untouched when it is not made of citations only,
e.g. ``[the @foo docs]``,
when it cites a key missing from the bibliography (a warning is logged),
when it is the text of a link, e.g. ``[@user](https://example.org)``,
or when it is in inline code, e.g. a backquoted ``[@property]``.
The citation blocks in fenced code blocks are left untouched too,
as these blocks are stashed by ``pymdownx.superfences`` beforehand.
"""

from __future__ import annotations

import logging
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
    from collections.abc import Callable

    from markdown import Markdown
    from pybtex.database import BibliographyData

logger = logging.getLogger(__name__)

citation_block_regex = re.compile(r"\[([^\[\]]*)\](?![(\[])")
"""The regular expression to find a citation block, e.g. ``[@foo; @bar]``.

A bracket followed by a parenthesis or a bracket is a link, not a citation block.
"""

inline_code_regex = re.compile(r"(`+).+?\1")
"""The regular expression to find an inline code span, e.g. a backquoted ``[@foo]``."""

citation_regex = re.compile(
    r"\s*(?:(?P<prefix>[^@;,]*?)\s+)?@(?P<key>[\w-]+)(?:,\s*(?P<suffix>[^@;]+))?\s*"
)
"""The regular expression that a whole citation must match, e.g. ``see @foo, p. 1``.

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


class _CitationPreprocessor(Preprocessor):
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
            matches = [citation_regex.fullmatch(part) for part in parts]
            if not parts or None in matches:
                return match.group(0)

            keys = [citation.group("key") for citation in matches]
            unknown_keys = [
                key for key in keys if key not in self.__bibliography.entries
            ]
            if unknown_keys:
                logger.warning(
                    "The citation keys %s are not in the bibliography.", unknown_keys
                )
                return match.group(0)

            cited_keys.update(dict.fromkeys(keys))
            return "".join(f"[^cite-{key}]" for key in keys)

        lines = [self.__replace_outside_code(replace, line) for line in lines]
        if cited_keys:
            lines.append("")
            lines.extend(f"[^cite-{key}]: {self.__format(key)}" for key in cited_keys)

        return lines

    @staticmethod
    def __replace_outside_code(
        replace: Callable[[re.Match[str]], str], line: str
    ) -> str:
        """Replace the citation blocks of a line that are not in inline code.

        Args:
            replace: The function replacing a citation block.
            line: The line.

        Returns:
            The line with the citation blocks replaced.
        """
        new_line = []
        start = 0
        for code in inline_code_regex.finditer(line):
            new_line.extend((
                citation_block_regex.sub(replace, line[start : code.start()]),
                code.group(0),
            ))
            start = code.end()

        new_line.append(citation_block_regex.sub(replace, line[start:]))
        return "".join(new_line)

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
            _CitationPreprocessor(md, bibliography), "bibtex_citations", self.priority
        )


def makeExtension(**kwargs: Any) -> CitationExtension:  # noqa: N802
    """Create the extension.

    Args:
        **kwargs: The configuration of the extension.

    Returns:
        The extension.
    """
    return CitationExtension(**kwargs)
