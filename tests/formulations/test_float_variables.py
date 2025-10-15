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
from __future__ import annotations  # noqa: I001


import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.auto_py import AutoPyDiscipline
from numpy import array
from numpy import zeros

from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario
from numpy import ndarray  # noqa: TC002


ZERO = zeros(1)


def f_float(x: float = 0.0, u: float = 0.0) -> float:
    y = x + u
    return y  # noqa: RET504


def f_ndarray(x: ndarray = ZERO, u: ndarray = ZERO) -> ndarray:
    y = x + u
    return y  # noqa: RET504


@pytest.fixture
def design_space() -> DesignSpace:
    """The design space."""
    design_space_ = DesignSpace()
    design_space_.add_variable("x", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    return design_space_


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace():
    """The uncertain space."""
    uncertain_space_ = ParameterSpace()
    uncertain_space_.add_random_variable("u", "OTNormalDistribution")
    return uncertain_space_


def test_float_variables(uncertain_space, design_space, statistic_estimation_settings):
    """Check that UScenario can handle float variables."""
    initial_current_value = design_space.get_current_value()
    discipline = AutoPyDiscipline(f_ndarray)
    umdo_scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="MDF",
        statistic_estimation_settings=statistic_estimation_settings,
    )
    umdo_scenario.execute(algo_name="CustomDOE", samples=array([[0.0]]))
    reference_f_opt = umdo_scenario.optimization_result.f_opt

    design_space.set_current_value(initial_current_value)
    discipline = AutoPyDiscipline(f_float)
    umdo_scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="MDF",
        statistic_estimation_settings=statistic_estimation_settings,
    )
    umdo_scenario.execute(algo_name="CustomDOE", samples=array([[0.0]]))

    assert umdo_scenario.optimization_result.f_opt == reference_f_opt
