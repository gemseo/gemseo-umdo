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
"""Fixtures for the tests of the U-MDO scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import SPNormalDistribution_Settings

if TYPE_CHECKING:
    from gemseo.discipline import Discipline


@pytest.fixture
def disciplines() -> list[Discipline]:
    """Three simple disciplines."""
    disc0 = AnalyticDiscipline(
        {"f": "x0+y1+y2", "c": "x0+y1+y2", "o": "x0+y1+y2"}, name="D0"
    )
    disc1 = AnalyticDiscipline({"y1": "x0+x1+2*y2"}, name="D1")
    disc2 = AnalyticDiscipline({"y2": "x0+x2+y1+u"}, name="D2")
    return [disc0, disc1, disc2]


@pytest.fixture
def design_space() -> DesignSpace:
    """The space of local and global design variables."""
    space = DesignSpace()
    for name in ["x0", "x1", "x2"]:
        space.add_variable(name, lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("y1", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return space


@pytest.fixture
def uncertain_space() -> RandomSpace:
    """The space defining the uncertain variable."""
    space = RandomSpace()
    space.add_variable("u", SPNormalDistribution_Settings())
    return space
