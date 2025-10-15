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
"""Compatibility between different versions of numpy."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING
from typing import Final

from packaging.version import parse as parse_version

if TYPE_CHECKING:
    from packaging.version import Version

NP_VERSION: Final[Version] = parse_version(version("numpy"))

if parse_version("2.0.0") <= NP_VERSION:
    from numpy import trapezoid  # noqa: F401

else:  # pragma: no cover
    from numpy import trapz as trapezoid  # noqa: F401
