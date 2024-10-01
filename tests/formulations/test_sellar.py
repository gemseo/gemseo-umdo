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
"""Test the different U-UMDO formulations with the Sellar's MDO problem."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.scenarios.doe_scenario import DOEScenario
from numpy.testing import assert_almost_equal

from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario

if TYPE_CHECKING:
    from gemseo.typing import RealArray


@pytest.fixture(scope="module", params=[1, 2])
def size(request) -> int:
    """The size of the coupling variables in the Sellar's problem."""
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def maximize_objective(request) -> bool:
    """Whether to maximize the objective."""
    return request.param


@pytest.fixture(scope="module")
def scenario_input_data() -> dict[str, str | int]:
    """The input data of the scenario."""
    return {"algo": "OT_MONTE_CARLO", "n_samples": 4}


@pytest.fixture(scope="module")
def disciplines(size) -> tuple[Sellar1, Sellar2, SellarSystem]:
    """The disciplines of the Sellar's problem."""
    return Sellar1(n=size), Sellar2(n=size), SellarSystem(n=size)


@pytest.fixture(scope="module")
def design_space(size) -> SellarDesignSpace:
    """The design space of the Sellar's problem."""
    return SellarDesignSpace(n=size)


@pytest.fixture(scope="module")
def reference_data(
    disciplines, design_space, maximize_objective, scenario_input_data
) -> RealArray:
    """The reference data.

    Monte Carlo samples of the Sellar's multidisciplinary system orchestrated by MDF.
    """
    doe_scenario = DOEScenario(
        disciplines,
        "MDF",
        "obj",
        design_space,
        maximize_objective=maximize_objective,
        max_mda_iter=3,
    )
    doe_scenario.add_constraint("c_1", "ineq")
    doe_scenario.add_constraint("c_2", "ineq")
    doe_scenario.execute(scenario_input_data)
    return doe_scenario.to_dataset().to_numpy()


@pytest.fixture(scope="module")
def dirac_uncertain_space() -> ParameterSpace:
    """An uncertain space for the Sellar's U-MDO problem with Dirac distributions."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable(
        "alpha", "OTDiracDistribution", variable_value=3.16
    )
    parameter_space.add_random_variable(
        "beta", "OTDiracDistribution", variable_value=24.0
    )
    parameter_space.add_random_variable(
        "gamma", "OTDiracDistribution", variable_value=0.2
    )
    return parameter_space


@pytest.fixture(scope="module")
def normal_uncertain_space() -> ParameterSpace:
    """An uncertain space for the Sellar's U-MDO problem with normal distributions."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable(
        "alpha", "OTNormalDistribution", mu=3.16, sigma=1e-6
    )
    parameter_space.add_random_variable(
        "beta", "OTNormalDistribution", mu=24.0, sigma=1e-6
    )
    parameter_space.add_random_variable(
        "gamma", "OTNormalDistribution", mu=0.2, sigma=1e-6
    )
    return parameter_space


statistic_estimation_settings = [
    ("Sampling", {"n_samples": 10}),
    ("Sampling", {"n_samples": 10, "estimate_statistics_iteratively": False}),
    ("SequentialSampling", {"n_samples": 10}),
    (
        "SequentialSampling",
        {"n_samples": 10, "estimate_statistics_iteratively": False},
    ),
    ("TaylorPolynomial", {}),
    ("TaylorPolynomial", {"second_order": True}),
]


@pytest.mark.parametrize(
    ("statistic_estimation", "statistic_estimation_parameters"),
    statistic_estimation_settings,
)
def test_uncertainty_free(
    disciplines,
    design_space,
    dirac_uncertain_space,
    reference_data,
    statistic_estimation,
    statistic_estimation_parameters,
    maximize_objective,
    scenario_input_data,
):
    """Check that the UDOEScenario and DOEScenario give the same results.

    For that, we model alpha, beta and gamma as random variables distributed according
    to Dirac distributions, so that all their realizations are equal to the
    corresponding alpha, beta and gamma in the uncertainty-free original problem.

    Some U-MDO formulations cannot be considered:

    - PCE because a PCE regressor cannot be trained with constant inputs,
    - ControlVariate because control variate estimators require non-constant inputs.
    """
    u_doe_scenario = UDOEScenario(
        disciplines,
        "MDF",
        "obj",
        design_space,
        dirac_uncertain_space,
        "Mean",
        maximize_objective=maximize_objective,
        statistic_estimation=statistic_estimation,
        statistic_estimation_parameters=statistic_estimation_parameters,
        max_mda_iter=3,
    )
    u_doe_scenario.add_constraint("c_1", "Mean")
    u_doe_scenario.add_constraint("c_2", "Mean")
    u_doe_scenario.execute(scenario_input_data)
    assert_almost_equal(u_doe_scenario.to_dataset().to_numpy(), reference_data)


@pytest.mark.parametrize(
    ("statistic_estimation", "statistic_estimation_parameters"),
    [
        *statistic_estimation_settings,
        ("ControlVariate", {"n_samples": 10}),
        ("PCE", {"doe_n_samples": 20}),
    ],
)
def test_weak_uncertainties(
    disciplines,
    design_space,
    normal_uncertain_space,
    reference_data,
    statistic_estimation,
    statistic_estimation_parameters,
    maximize_objective,
    scenario_input_data,
):
    """Check that the UDOEScenario and DOEScenario give the same results.

    For that, we model alpha, beta and gamma as random variables distributed according
    to normal distributions with a small variance.
    """
    u_doe_scenario = UDOEScenario(
        disciplines,
        "MDF",
        "obj",
        design_space,
        normal_uncertain_space,
        "Mean",
        maximize_objective=maximize_objective,
        statistic_estimation=statistic_estimation,
        statistic_estimation_parameters=statistic_estimation_parameters,
        max_mda_iter=3,
    )
    u_doe_scenario.add_constraint("c_1", "Mean")
    u_doe_scenario.add_constraint("c_2", "Mean")
    u_doe_scenario.execute(scenario_input_data)
    data = u_doe_scenario.to_dataset().to_numpy()
    assert_almost_equal(data, reference_data, decimal=5)
