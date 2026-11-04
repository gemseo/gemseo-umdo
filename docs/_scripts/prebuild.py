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
"""Generate the documentation files that Zensical cannot generate itself.

Zensical supports neither MkDocs hooks nor the mkdocs-gallery, gen-files
and markdown-exec plugins, so this script does their work before the build.
Run it from the root of the repository before building or serving the documentation:

    uv run --group doc python docs/_scripts/prebuild.py

It

- runs the mkdocs-gallery examples into ``docs/generated/examples``,
- writes ``_docs/generated.yml``, inherited by ``mkdocs.yml``,
  with the navigation of ``nav.yml`` where the gallery directories are expanded,
- writes the credits into ``_docs/credits.md``.

An example failing unexpectedly makes the script fail.

An example runs again only when its file changes,
as mkdocs-gallery compares its MD5 with the one stored in ``docs/generated/examples``;
delete this directory to run all the examples again,
e.g. after a change of the library that changes their outputs.
Set the environment variable ``DOC_PLOT_GALLERY`` to ``0``
to write the example pages without running the examples,
for a fast build of the other pages.
"""

from __future__ import annotations

import os
import re
import runpy
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import yaml
from mkdocs_gallery import glr_path_static
from mkdocs_gallery.gen_gallery import fill_mkdocs_nav
from mkdocs_gallery.gen_gallery import generate_gallery_md
from mkdocs_gallery.gen_gallery import parse_config
from mkdocs_gallery.gen_gallery import summarize_failing_examples

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

root_dir = Path.cwd()
"""The root directory of the repository."""

docs_dir = root_dir / "docs"
"""The directory of the documentation sources."""

generated_dir = root_dir / "_docs"
"""The directory of the generated files that are not documentation pages."""

_relative_image = re.compile(r'((?:src|srcset)=")\.\./images/')


def run_gallery() -> dict[Path, Any]:
    """Run the gallery examples and write their pages.

    Returns:
        The table of contents of each gallery.
    """
    mkdocs_conf = {
        "config_file_path": str(root_dir / "mkdocs.yml"),
        "docs_dir": str(docs_dir),
        "site_dir": str(root_dir / "site"),
    }
    gallery_conf = parse_config(
        {
            "conf_script": str(docs_dir / "gallery_conf.py"),
            "default_thumb_file": str(docs_dir / "images" / "monogram.png"),
            "plot_gallery": os.environ.get("DOC_PLOT_GALLERY", "1") != "0",
        },
        mkdocs_conf,
    )
    galleries_tocs, _ = generate_gallery_md(gallery_conf, mkdocs_conf)
    summarize_failing_examples(gallery_conf, mkdocs_conf)
    return galleries_tocs


def fix_image_paths() -> None:
    """Make the image paths of the multi-figure examples relative to their page file.

    mkdocs-gallery writes them relative to the page URL (``../images/``)
    in raw HTML, whereas Zensical resolves them relative to the page file.
    """
    for path in (docs_dir / "generated").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        new_text = _relative_image.sub(r"\1images/", text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")


def copy_gallery_stylesheets() -> None:
    """Copy the stylesheets of mkdocs-gallery into the documentation assets."""
    output_dir = docs_dir / "css" / "gallery"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in Path(glr_path_static()).glob("*.css"):
        shutil.copy(path, output_dir / path.name)


def remove_empty_sections(nav: Iterable[Any]) -> list[Any]:
    """Remove the sections without pages from a navigation.

    The tables of contents of mkdocs-gallery include a section without pages
    for the title of each gallery index,
    which MkDocs used to drop but Zensical displays.

    Args:
        nav: The navigation.

    Returns:
        The navigation without empty sections.
    """
    new_nav = []
    for item in nav:
        if isinstance(item, dict):
            ((title, value),) = item.items()
            if isinstance(value, list):
                value = remove_empty_sections(value)
                if not value:
                    continue

            item = {title: value}

        new_nav.append(item)

    return new_nav


def write_generated_config(galleries_tocs: Mapping[Path, Any]) -> None:
    """Write the configuration generated from the gallery.

    Args:
        galleries_tocs: The table of contents of each gallery.
    """
    nav = yaml.safe_load((root_dir / "nav.yml").read_text(encoding="utf-8"))["nav"]
    config = {
        "nav": remove_empty_sections(
            fill_mkdocs_nav({"docs_dir": str(docs_dir), "nav": nav}, galleries_tocs)
        ),
    }
    generated_dir.mkdir(exist_ok=True)
    (generated_dir / "generated.yml").write_text(
        f"# Generated by docs/_scripts/prebuild.py, do not edit.\n"
        f"{yaml.safe_dump(config, allow_unicode=True, sort_keys=False)}",
        encoding="utf-8",
    )


def main() -> None:
    """Generate the documentation files."""
    galleries_tocs = run_gallery()
    fix_image_paths()
    copy_gallery_stylesheets()
    write_generated_config(galleries_tocs)
    render_credits = runpy.run_path(str(docs_dir / "_scripts" / "credits.py"))[
        "render_credits"
    ]
    (generated_dir / "credits.md").write_text(render_credits(), encoding="utf-8")


if __name__ == "__main__":
    main()
