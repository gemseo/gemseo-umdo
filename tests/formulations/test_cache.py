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
"""Tests for the cache of the U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import CustomDOE_Settings
from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.formulation import DisciplinaryOpt_Settings
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from numpy import array

from gemseo_umdo.formulations.control_variate_settings import ControlVariate_Settings
from gemseo_umdo.formulations.pce_settings import PCE_Settings
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)
from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo_umdo.formulations.base_umdo_formulation_settings import (
        BaseUMDOFormulationSettings,
    )

DESIGN_POINTS = array([[-1.0], [0.0], [1.0]])
"""The design points at which to estimate the statistics."""


def create_scenario(
    settings: BaseUMDOFormulationSettings, add_constraints: bool
) -> UMDOScenario:
    """Create a U-MDO scenario with a discipline computing an objective and two outputs.

    Args:
        settings: The settings of the U-MDO formulation.
        add_constraints: Whether to add two margin constraints.

    Returns:
        The scenario.
    """
    discipline = AnalyticDiscipline({
        "f": "(x-1)**2+u",
        "g1": "x+u-3",
        "g2": "-x+u-3",
    })
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-3.0, upper_bound=3.0, value=0.0)
    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", OTNormalDistribution_Settings())
    scenario = UMDOScenario(
        [discipline],
        design_space,
        uncertain_space,
        statistic_estimation_settings=settings,
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    scenario.add_objective("f", "Mean")
    if add_constraints:
        scenario.add_constraint("g1", "Margin")
        scenario.add_constraint("g2", "Margin")
    return scenario


def create_sequential_sampling_settings(
    estimate_statistics_iteratively: bool = True,
) -> SequentialSampling_Settings:
    """Create the settings of a SequentialSampling formulation.

    Args:
        estimate_statistics_iteratively: Whether to estimate the statistics
            iteratively.

    Returns:
        The settings of a SequentialSampling formulation.
    """
    return SequentialSampling_Settings(
        doe_algo_settings=OT_OPT_LHS_Settings(n_samples=7),
        initial_n_samples=3,
        n_samples_increment=2,
        estimate_statistics_iteratively=estimate_statistics_iteratively,
    )


@pytest.mark.parametrize(
    ("create_settings", "eval_jac"),
    [
        pytest.param(
            lambda: Sampling_Settings(n_samples=5), False, id="Sampling-iterative"
        ),
        pytest.param(
            lambda: Sampling_Settings(n_samples=5), True, id="Sampling-iterative-jac"
        ),
        pytest.param(
            lambda: Sampling_Settings(
                n_samples=5, estimate_statistics_iteratively=False
            ),
            False,
            id="Sampling",
        ),
        pytest.param(
            lambda: Sampling_Settings(
                n_samples=5, estimate_statistics_iteratively=False
            ),
            True,
            id="Sampling-jac",
        ),
        pytest.param(
            create_sequential_sampling_settings,
            False,
            id="SequentialSampling-iterative",
        ),
        pytest.param(
            create_sequential_sampling_settings,
            True,
            id="SequentialSampling-iterative-jac",
        ),
        pytest.param(
            lambda: create_sequential_sampling_settings(False),
            False,
            id="SequentialSampling",
        ),
        pytest.param(
            lambda: create_sequential_sampling_settings(False),
            True,
            id="SequentialSampling-jac",
        ),
        pytest.param(
            lambda: ControlVariate_Settings(n_samples=5), False, id="ControlVariate"
        ),
        pytest.param(lambda: PCE_Settings(n_samples=5), False, id="PCE"),
        pytest.param(lambda: Surrogate_Settings(n_samples=5), False, id="Surrogate"),
    ],
)
@pytest.mark.parametrize("add_constraints", [False, True])
def test_one_sampling_per_design_point(
    create_settings: Callable[[], BaseUMDOFormulationSettings],
    eval_jac: bool,
    add_constraints: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that the disciplines are sampled once per design point.

    The constraints must not trigger new samplings,
    while the Jacobians require a second sampling
    and must be estimated at each design point.
    """
    scenario = create_scenario(create_settings(), add_constraints)
    formulation = scenario.formulation
    compute_samples = formulation.compute_samples
    n_calls = 0

    def count_compute_samples(*args: Any, **kwargs: Any) -> Any:
        nonlocal n_calls
        n_calls += 1
        return compute_samples(*args, **kwargs)

    monkeypatch.setattr(formulation, "compute_samples", count_compute_samples)
    scenario.execute(CustomDOE_Settings(samples=DESIGN_POINTS, eval_jac=eval_jac))
    assert n_calls == len(DESIGN_POINTS) * (1 + int(eval_jac))
    if eval_jac:
        problem = formulation.problem
        for function in problem.functions:
            gradients = problem.database.get_gradient_history(function.name)
            assert len(gradients) == len(DESIGN_POINTS)
