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
from __future__ import annotations

from pathlib import Path

file_dir_path = Path(__file__).parent
example_dir_name = "examples"
gallery_dir = file_dir_path / "generated" / example_dir_name
examples_dir = file_dir_path / example_dir_name
examples_subdirs = [
    subdir.name
    for subdir in examples_dir.iterdir()
    if (examples_dir / subdir).is_dir()
    and (examples_dir / subdir / "README.md").is_file()
]


def _patch_py_source_parser_for_py314():
    # mkdocs-gallery uses removed `ast.Str` and `Constant.s`. Replace
    # `_get_docstring_and_rest` with a Constant-based equivalent.
    import ast
    import platform
    import tokenize
    from io import BytesIO

    from mkdocs_gallery import py_source_parser
    from mkdocs_gallery.errors import ExtensionError
    from packaging.version import parse as parse_version

    def _get_docstring_and_rest(file):
        node, content = py_source_parser.parse_source_file(file)
        if node is None:
            return py_source_parser.SYNTAX_ERROR_DOCSTRING, content, 1, node
        if not isinstance(node, ast.Module):
            msg = (
                "This function only supports modules. "
                f"You provided {node.__class__.__name__}"
            )
            raise ExtensionError(msg)
        is_str_const = (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        )
        if not is_str_const:
            msg = (
                f'Could not find docstring in file "{file}". '
                "A docstring is required by mkdocs-gallery "
                'unless the file is ignored by "ignore_pattern"'
            )
            raise ExtensionError(msg)
        if parse_version(platform.python_version()) >= parse_version("3.7"):
            docstring = ast.get_docstring(node)
            assert docstring is not None
            raw = node.body[0].value.value
            if len(raw) and raw[0] == "\n":
                docstring = "\n" + docstring
            ts = tokenize.tokenize(BytesIO(content.encode()).readline)
            for tk in ts:
                if tk.exact_type == 3:
                    lineno, _ = tk.end
                    break
            else:
                lineno = 0
        else:
            docstring_node = node.body[0]
            docstring = docstring_node.value.value
            lineno = docstring_node.lineno
        rest = "\n".join(content.split("\n")[lineno:])
        lineno += 1
        return docstring, rest, lineno, node

    py_source_parser._get_docstring_and_rest = _get_docstring_and_rest


_patch_py_source_parser_for_py314()

conf = {
    f"{example_dir_name}_dirs": [examples_dir / subdir for subdir in examples_subdirs],
    "gallery_dirs": [gallery_dir / subdir for subdir in examples_subdirs],
}
