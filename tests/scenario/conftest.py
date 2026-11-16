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

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import SPNormalDistribution_Settings


@pytest.fixture
def toy_problem() -> tuple[AnalyticDiscipline, DesignSpace, RandomSpace]:
    """A discipline computing y=(x+u)**2 and z=x, its design and uncertain spaces."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1.0, upper_bound=1.0, value=0.5)
    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", SPNormalDistribution_Settings())
    discipline = AnalyticDiscipline({"y": "(x+u)**2", "z": "x"})
    return discipline, design_space, uncertain_space
