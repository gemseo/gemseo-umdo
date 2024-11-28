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

from gemseo_umdo.formulations.control_variate_settings import ControlVariate_Settings
from gemseo_umdo.formulations.pce_settings import PCE_Settings
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)
from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings
from gemseo_umdo.formulations.taylor_polynomial_settings import (
    TaylorPolynomial_Settings,
)
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
    return {"algo_name": "OT_MONTE_CARLO", "n_samples": 4}


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
        "obj",
        design_space,
        formulation_name="MDF",
        maximize_objective=maximize_objective,
        main_mda_settings={"max_mda_iter": 3},
    )
    doe_scenario.add_constraint("c_1", "ineq")
    doe_scenario.add_constraint("c_2", "ineq")
    doe_scenario.execute(**scenario_input_data)
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
    (Sampling_Settings, {"n_samples": 10}),
    (Sampling_Settings, {"n_samples": 10, "estimate_statistics_iteratively": False}),
    (SequentialSampling_Settings, {"n_samples": 10}),
    (
        SequentialSampling_Settings,
        {"n_samples": 10, "estimate_statistics_iteratively": False},
    ),
    (TaylorPolynomial_Settings, {}),
    (TaylorPolynomial_Settings, {"second_order": True}),
]


@pytest.mark.parametrize(
    "statistic_estimation_settings",
    statistic_estimation_settings,
)
def test_uncertainty_free(
    disciplines,
    design_space,
    dirac_uncertain_space,
    reference_data,
    statistic_estimation_settings,
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
    cls, kwargs = statistic_estimation_settings
    u_doe_scenario = UDOEScenario(
        disciplines,
        "obj",
        design_space,
        dirac_uncertain_space,
        "Mean",
        formulation_name="MDF",
        maximize_objective=maximize_objective,
        statistic_estimation_settings=cls(**kwargs),
        main_mda_settings={"max_mda_iter": 3},
    )
    u_doe_scenario.add_constraint("c_1", "Mean")
    u_doe_scenario.add_constraint("c_2", "Mean")
    u_doe_scenario.execute(**scenario_input_data)
    assert_almost_equal(u_doe_scenario.to_dataset().to_numpy(), reference_data)


@pytest.mark.parametrize(
    "statistic_estimation_settings",
    [
        *statistic_estimation_settings,
        (ControlVariate_Settings, {"n_samples": 10}),
        (PCE_Settings, {"n_samples": 20}),
        (Surrogate_Settings, {"n_samples": 20}),
    ],
)
def test_weak_uncertainties(
    disciplines,
    design_space,
    normal_uncertain_space,
    reference_data,
    statistic_estimation_settings,
    maximize_objective,
    scenario_input_data,
):
    """Check that the UDOEScenario and DOEScenario give the same results.

    For that, we model alpha, beta and gamma as random variables distributed according
    to normal distributions with a small variance.
    """
    cls, kwargs = statistic_estimation_settings
    u_doe_scenario = UDOEScenario(
        disciplines,
        "obj",
        design_space,
        normal_uncertain_space,
        "Mean",
        formulation_name="MDF",
        maximize_objective=maximize_objective,
        statistic_estimation_settings=cls(**kwargs),
        main_mda_settings={"max_mda_iter": 3},
    )
    u_doe_scenario.add_constraint("c_1", "Mean")
    u_doe_scenario.add_constraint("c_2", "Mean")
    u_doe_scenario.execute(**scenario_input_data)
    data = u_doe_scenario.to_dataset().to_numpy()
    assert_almost_equal(data, reference_data, decimal=5)
