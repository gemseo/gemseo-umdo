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
"""Scenario for multidisciplinary design sampling problems under uncertainty."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from gemseo.core.doe_scenario import DOEScenario

from gemseo_umdo.scenarios._scenario import _UScenario


class UDOEScenario(_UScenario, DOEScenario):
    """A DOE-based scenario for multidisciplinary design under uncertainty."""

    GRAMMAR_DIRECTORY: ClassVar[Path] = Path(__file__) / "udoe"
