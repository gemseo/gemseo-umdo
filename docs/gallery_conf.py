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

gallery_dir = Path(__file__).parent / "generated" / "examples"
examples_dir = Path(__file__).parent / "examples"
examples_subdirs = [
    subdir.name
    for subdir in examples_dir.iterdir()
    if (examples_dir / subdir).is_dir()
    and (examples_dir / subdir / "README.md").is_file()
]
examples_dirs = [examples_dir / subdir for subdir in examples_subdirs]
gallery_dirs = [gallery_dir / subdir for subdir in examples_subdirs]

conf = {"examples_dirs": examples_dirs, "gallery_dirs": gallery_dirs}
