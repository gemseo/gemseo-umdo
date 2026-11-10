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

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import SPNormalDistribution_Settings
from gemseo.util.testing.pytest_conftest import *  # noqa: F403


@pytest.fixture
def toy_disciplines() -> list[AnalyticDiscipline]:
    """Three coupled disciplines of a toy problem, with two strongly coupled ones."""
    disc0 = AnalyticDiscipline(
        {"f": "x0+y1+y2", "c": "x0+y1+y2", "o": "x0+y1+y2"}, name="D0"
    )
    disc1 = AnalyticDiscipline({"y1": "x0+x1+2*y2"}, name="D1")
    disc2 = AnalyticDiscipline({"y2": "x0+x2+y1+u"}, name="D2")
    return [disc0, disc1, disc2]


@pytest.fixture
def toy_design_space() -> DesignSpace:
    """The design space of the toy problem."""
    space = DesignSpace()
    for name in ["x0", "x1", "x2"]:
        space.add_variable(name, lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("y1", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return space


@pytest.fixture
def toy_uncertain_space(request) -> RandomSpace:
    """The uncertain space of the toy problem, with a standard normal variable `u`.

    The distribution settings can be passed by indirect parametrization,
    SciPy-based by default.
    """
    space = RandomSpace()
    space.add_variable("u", getattr(request, "param", SPNormalDistribution_Settings()))
    return space
