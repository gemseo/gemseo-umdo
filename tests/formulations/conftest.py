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

from typing import Sequence

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.chain import MDOChain
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.mdf import MDF


@pytest.fixture
def disciplines() -> list[AnalyticDiscipline]:
    """The coupled disciplines."""
    disc0 = AnalyticDiscipline(
        {"f": "x0+y1+y2+u", "c": "x0+y1+y2+2*u", "o": "x0+y1+y2+3*u"}, name="D0"
    )
    disc1 = AnalyticDiscipline({"y1": "x0+x1+2*y2+u1"}, name="D1")
    disc2 = AnalyticDiscipline({"y2": "x0+x2+y1+u2"}, name="D2")
    return [disc0, disc1, disc2]


@pytest.fixture
def mdf_discipline() -> MDOChain:
    """A monodisciplinary version of ``disciplines``."""
    disc0 = AnalyticDiscipline(
        {"f": "x0+y1+y2+u", "c": "x0+y1+y2+2*u", "o": "x0+y1+y2+3*u"}, name="D0"
    )
    disc1 = AnalyticDiscipline({"y1": "-(3*x0+x1+2*x2+u1+2*u2)"}, name="D1")
    disc2 = AnalyticDiscipline({"y2": "-(2*x0+x1+x2+u1+u2)"}, name="D2")
    return MDOChain([disc1, disc2, disc0])


@pytest.fixture
def design_space() -> DesignSpace:
    """The design space."""
    space = DesignSpace()
    space.add_variable("x0", l_b=0.0, u_b=1.0, value=0.5)
    space.add_variable("x1", l_b=0.0, u_b=2.0, value=0.5)
    space.add_variable("x2", l_b=0.0, u_b=3.0, value=0.5)
    return space


@pytest.fixture
def uncertain_space() -> ParameterSpace:
    """The uncertain space."""
    space = ParameterSpace()
    space.add_random_variable("u", "SPNormalDistribution", mu=1.0, sigma=1.0)
    space.add_random_variable("u1", "SPNormalDistribution", mu=2.0, sigma=2.0)
    space.add_random_variable("u2", "SPNormalDistribution", mu=3.0, sigma=3.0)
    return space


@pytest.fixture
def mdo_formulation(
    disciplines: Sequence[MDODiscipline], uncertain_space: ParameterSpace
) -> MDF:
    """The MDO formulation."""
    return MDF(disciplines, "f", uncertain_space)
