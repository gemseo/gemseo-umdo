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
"""Compatibility between different versions of openturns."""

from __future__ import annotations

from importlib.metadata import version
from typing import Final

import openturns
from packaging.version import Version
from packaging.version import parse as parse_version

OT_VERSION: Final[Version] = parse_version(version("openturns"))

if parse_version("1.22.0") <= OT_VERSION:

    def IterativeThresholdExceedance(dimension, threshold):  # noqa:D103, N802
        return openturns.IterativeThresholdExceedance(
            dimension, openturns.Greater(), threshold
        )

else:

    def IterativeThresholdExceedance(dimension, threshold):  # noqa:D103, N802
        return openturns.IterativeThresholdExceedance(dimension, threshold)
