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
from typing import Final
from typing import TypeAlias
from unittest import mock

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import CustomDOE_Settings
from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.formulation import DisciplinaryOpt_Settings
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.util.testing.helper import assert_exception
from numpy import array
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo_umdo.formulation._function.pce import StatisticFunctionForPCE
from gemseo_umdo.formulation.control_variate_settings import ControlVariate_Settings
from gemseo_umdo.formulation.pce_settings import PCE_Settings
from gemseo_umdo.formulation.sampling import Sampling
from gemseo_umdo.formulation.sampling_settings import Sampling_Settings
from gemseo_umdo.formulation.sequential_sampling_settings import (
    SequentialSampling_Settings,
)
from gemseo_umdo.formulation.surrogate_settings import Surrogate_Settings
from gemseo_umdo.scenario.mdo import UMDOScenario

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from gemseo.util.typing import RealArray

    from gemseo_umdo.formulation.core.base_settings import BaseUMDOFormulationSettings

design_points: Final[RealArray] = array([[-1.0], [0.0], [1.0]])
"""The design points at which to estimate the statistics."""

JacobianRequest: TypeAlias = tuple[
    UMDOScenario, "Callable[[RealArray], RealArray]", "RealArray"
]
"""A scenario, the Jacobian function of its objective and a design point."""


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
    ("create_settings", "eval_jac", "n_executions", "n_linearizations"),
    [
        pytest.param(
            lambda: Sampling_Settings(n_samples=5),
            False,
            15,
            0,
            id="Sampling-iterative",
        ),
        pytest.param(
            lambda: Sampling_Settings(n_samples=5),
            True,
            15,
            15,
            id="Sampling-iterative-jac",
        ),
        pytest.param(
            lambda: Sampling_Settings(
                n_samples=5, estimate_statistics_iteratively=False
            ),
            False,
            15,
            0,
            id="Sampling",
        ),
        pytest.param(
            lambda: Sampling_Settings(
                n_samples=5, estimate_statistics_iteratively=False
            ),
            True,
            15,
            15,
            id="Sampling-jac",
        ),
        pytest.param(
            create_sequential_sampling_settings,
            False,
            15,
            0,
            id="SequentialSampling-iterative",
        ),
        pytest.param(
            create_sequential_sampling_settings,
            True,
            15,
            15,
            id="SequentialSampling-iterative-jac",
        ),
        pytest.param(
            lambda: create_sequential_sampling_settings(False),
            False,
            15,
            0,
            id="SequentialSampling",
        ),
        pytest.param(
            lambda: create_sequential_sampling_settings(False),
            True,
            15,
            15,
            id="SequentialSampling-jac",
        ),
        # ControlVariate also executes and linearizes the disciplines
        # at the mean of the uncertain variables.
        pytest.param(
            lambda: ControlVariate_Settings(n_samples=5),
            False,
            18,
            3,
            id="ControlVariate",
        ),
        pytest.param(lambda: PCE_Settings(n_samples=5), False, 15, 0, id="PCE"),
        pytest.param(lambda: PCE_Settings(n_samples=5), True, 15, 15, id="PCE-jac"),
        pytest.param(
            lambda: Surrogate_Settings(n_samples=5), False, 15, 0, id="Surrogate"
        ),
    ],
)
@pytest.mark.parametrize("add_constraints", [False, True])
def test_one_sampling_per_design_point(
    create_settings: Callable[[], BaseUMDOFormulationSettings],
    eval_jac: bool,
    n_executions: int,
    n_linearizations: int,
    add_constraints: bool,
    enable_discipline_statistics: None,
) -> None:
    """Check that the disciplines are sampled once per design point.

    Neither the constraints nor the Jacobians must trigger new samplings.
    The disciplines must be executed once per sample
    and linearized once per sample when the Jacobians are required;
    with SequentialSampling,
    the 3 design points are sampled with 3, 5 and 7 samples.
    """
    scenario = create_scenario(create_settings(), add_constraints)
    formulation = scenario.formulation
    with mock.patch.object(
        formulation, "compute_samples", wraps=formulation.compute_samples
    ) as compute_samples:
        scenario.execute(CustomDOE_Settings(samples=design_points, eval_jac=eval_jac))

    assert compute_samples.call_count == len(design_points)
    statistics = scenario.disciplines[0].execution_statistics
    assert statistics.n_executions == n_executions
    assert statistics.n_linearizations == n_linearizations


@pytest.mark.parametrize(
    "create_settings",
    [
        pytest.param(lambda: Sampling_Settings(n_samples=5), id="Sampling-iterative"),
        pytest.param(
            lambda: Sampling_Settings(
                n_samples=5, estimate_statistics_iteratively=False
            ),
            id="Sampling",
        ),
        pytest.param(
            create_sequential_sampling_settings, id="SequentialSampling-iterative"
        ),
        pytest.param(
            lambda: create_sequential_sampling_settings(False),
            id="SequentialSampling",
        ),
        pytest.param(lambda: PCE_Settings(n_samples=5), id="PCE"),
    ],
)
def test_gradients_at_each_design_point(
    create_settings: Callable[[], BaseUMDOFormulationSettings],
) -> None:
    """Check that the Jacobians of all the statistics are estimated at each point."""
    scenario = create_scenario(create_settings(), True)
    scenario.execute(CustomDOE_Settings(samples=design_points, eval_jac=True))
    problem = scenario.formulation.problem
    for function in problem.functions:
        gradients = problem.database.get_gradient_history(function.name)
        assert len(gradients) == len(design_points)


@pytest.mark.parametrize(
    ("create_settings", "n_executions"),
    [
        pytest.param(lambda: Sampling_Settings(n_samples=5), 20, id="Sampling"),
        # The first design point is sampled twice with 3 samples,
        # then the next ones with 5 and 7 samples.
        pytest.param(create_sequential_sampling_settings, 18, id="SequentialSampling"),
    ],
)
def test_jacobian_request_without_driver(
    create_settings: Callable[[], BaseUMDOFormulationSettings],
    n_executions: int,
    enable_discipline_statistics: None,
) -> None:
    """Check that a first Jacobian request enables the joint sampling.

    The first design point is sampled once for the statistics
    and once more, with the same number of samples, for their Jacobians;
    the next ones are sampled once for both.
    """
    scenario = create_scenario(create_settings(), True)
    formulation = scenario.formulation
    objective = formulation.problem.objective
    with mock.patch.object(
        formulation, "compute_samples", wraps=formulation.compute_samples
    ) as compute_samples:
        for design_point in design_points:
            objective.evaluate(design_point)
            objective.jac(design_point)

    assert compute_samples.call_count == len(design_points) + 1
    statistics = scenario.disciplines[0].execution_statistics
    assert statistics.n_executions == n_executions


@pytest.mark.parametrize(
    "estimate_statistics_iteratively",
    [pytest.param(True, id="iterative"), pytest.param(False, id="one-shot")],
)
def test_statistics_at_jacobian_resampling(
    estimate_statistics_iteratively: bool,
) -> None:
    """Check that resampling a point for the Jacobians keeps its statistics.

    With a seeded DOE algorithm,
    the resampling uses the same samples as the first sampling,
    so the constraint evaluated after the first Jacobian request
    has the value of the first sampling.
    """
    settings = Sampling_Settings(
        n_samples=5, estimate_statistics_iteratively=estimate_statistics_iteratively
    )
    design_point = design_points[0]
    other_scenario = create_scenario(settings.model_copy(deep=True), True)
    expected = other_scenario.formulation.problem.constraints[0].evaluate(design_point)
    problem = create_scenario(settings, True).formulation.problem
    problem.objective.evaluate(design_point)
    problem.objective.jac(design_point)
    assert_equal(problem.constraints[0].evaluate(design_point), expected)


def test_jacobian_after_statistics_only_resampling() -> None:
    """Check that a point resampled without the Jacobians is resampled for them.

    The point is first sampled for the statistics and their Jacobians,
    then for the statistics only,
    so that its cached data no longer allow to estimate the Jacobians.
    """
    formulation = create_scenario(Sampling_Settings(n_samples=5), False).formulation
    objective = formulation.problem.objective.original
    design_point = design_points[0]
    formulation.estimate_statistic_jacobians = True
    expected = objective.jac(design_point)
    formulation.reset_execution_state()
    formulation.input_data_to_output_data.clear()
    objective.evaluate(design_point)
    assert_equal(objective.jac(design_point), expected)


def test_sequential_sampling_schedule_at_same_point(
    enable_discipline_statistics: None,
) -> None:
    """Check that a new sampling of the last point uses the same number of samples.

    The number of samples is increased only at a new design point.
    """
    scenario = create_scenario(create_sequential_sampling_settings(), False)
    formulation = scenario.formulation
    objective = formulation.problem.objective
    design_point = design_points[0]
    objective.evaluate(design_point)
    formulation.input_data_to_output_data.clear()
    objective.evaluate(design_point)
    assert scenario.disciplines[0].execution_statistics.n_executions == 3 + 3


def create_scenario_with_older_point() -> UMDOScenario:
    """Create a SequentialSampling scenario evaluating its objective at two points.

    The first point is sampled with 3 samples and the second with 5.

    Returns:
        The scenario.
    """
    scenario = create_scenario(create_sequential_sampling_settings(), False)
    objective = scenario.formulation.problem.objective
    objective.evaluate(design_points[0])
    objective.evaluate(design_points[1])
    return scenario


@pytest.mark.parametrize(
    ("index", "n_samples"),
    [
        # A new design point increases the number of samples.
        pytest.param(0, 7, id="older_point"),
        # The last design point keeps its number of samples.
        pytest.param(1, 5, id="last_point"),
    ],
)
def test_sequential_sampling_statistics_again_with_jacobians(
    enable_discipline_statistics: None,
    index: int,
    n_samples: int,
) -> None:
    """Check the number of samples to estimate the statistics at a point again.

    The point is no longer cached
    and its statistics are estimated again with the joint sampling.
    """
    scenario = create_scenario_with_older_point()
    formulation = scenario.formulation
    formulation.input_data_to_output_data.clear()
    formulation.estimate_statistic_jacobians = True
    statistics = scenario.disciplines[0].execution_statistics
    n_executions = statistics.n_executions
    formulation.problem.objective.evaluate(design_points[index])
    assert statistics.n_executions - n_executions == n_samples


def test_sequential_sampling_jacobian_at_last_point_after_driver() -> None:
    """Check that the statistics and Jacobians at the last point use the same sampling.

    The last point of the execution is sampled again after this execution
    to estimate the Jacobians;
    as the statistic is the mean of x*u, its Jacobian is the mean of u,
    which must be the statistic divided by x.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=1.0, upper_bound=3.0, value=1.0)
    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", OTNormalDistribution_Settings())
    scenario = UMDOScenario(
        [AnalyticDiscipline({"f": "x*u"})],
        design_space,
        uncertain_space,
        statistic_estimation_settings=create_sequential_sampling_settings(),
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    scenario.add_objective("f", "Mean")
    design_points = array([[1.0], [2.0]])
    # The design points are sampled with 3 and 5 samples.
    scenario.execute(CustomDOE_Settings(samples=design_points))
    # The statistic function itself, without the database of the problem.
    objective = scenario.formulation.problem.objective.original
    design_point = design_points[-1]
    jacobian = objective.jac(design_point)
    assert_allclose(objective.evaluate(design_point), design_point * jacobian[0])


def test_sequential_sampling_joint_estimation_kept_after_driver() -> None:
    """Check that the statistics estimated with their Jacobians are kept.

    After an execution sampling the last point for both,
    evaluating the statistic and its Jacobian at this point,
    e.g. by a post-optimal analysis,
    must return the values of the database without a new sampling.
    """
    scenario = create_scenario(create_sequential_sampling_settings(), False)
    scenario.execute(CustomDOE_Settings(samples=design_points, eval_jac=True))
    formulation = scenario.formulation
    problem = formulation.problem
    # The statistic function itself, without the database of the problem.
    objective = problem.objective.original
    design_point = design_points[-1]
    with mock.patch.object(formulation, "compute_samples") as compute_samples:
        value = objective.evaluate(design_point)
        jacobian = objective.jac(design_point)

    compute_samples.assert_not_called()
    database = problem.database
    name = objective.name
    assert_equal(value, database.get_function_value(name, design_point))
    assert_equal(
        jacobian,
        database.get_function_value(database.get_gradient_name(name), design_point),
    )


def create_scenario_after_derivative_free_driver() -> UMDOScenario:
    """Create a SequentialSampling scenario executed without the Jacobians.

    The first two design points are sampled with 3 and 5 samples.

    Returns:
        The scenario.
    """
    scenario = create_scenario(create_sequential_sampling_settings(), False)
    scenario.execute(CustomDOE_Settings(samples=design_points[:2]))
    return scenario


def test_sequential_sampling_statistics_kept_after_driver() -> None:
    """Check that the statistics estimated without their Jacobians are kept.

    After an execution sampling the last point without the Jacobians,
    e.g. with a derivative-free optimizer,
    evaluating the statistic at this point,
    e.g. by a post-optimal analysis,
    must return the value of the database without a new sampling.
    """
    formulation = create_scenario_after_derivative_free_driver().formulation
    problem = formulation.problem
    # The statistic function itself, without the database of the problem.
    objective = problem.objective.original
    design_point = design_points[1]
    with mock.patch.object(formulation, "compute_samples") as compute_samples:
        value = objective.evaluate(design_point)

    compute_samples.assert_not_called()
    assert_equal(
        value, problem.database.get_function_value(objective.name, design_point)
    )


def prepare_older_point() -> JacobianRequest:
    """Prepare a Jacobian request at the older of two points sampled with 3 and 5.

    Returns:
        The scenario, the Jacobian function of its objective and the older point.
    """
    scenario = create_scenario_with_older_point()
    return scenario, scenario.formulation.problem.objective.jac, design_points[0]


def prepare_point_of_previous_execution() -> JacobianRequest:
    """Prepare a Jacobian request at the first point sampled by an execution.

    Returns:
        The scenario, the Jacobian function of its objective and the first point.
    """
    scenario = create_scenario(create_sequential_sampling_settings(), False)
    scenario.execute(CustomDOE_Settings(samples=design_points))
    return scenario, scenario.formulation.problem.objective.jac, design_points[0]


def prepare_last_point_after_derivative_free_driver() -> JacobianRequest:
    """Prepare a Jacobian request at the last point of a derivative-free execution.

    This point is sampled with 5 samples.

    Returns:
        The scenario, the Jacobian function of its objective and the last point.
    """
    scenario = create_scenario_after_derivative_free_driver()
    objective = scenario.formulation.problem.objective.original
    return scenario, objective.jac, design_points[1]


@pytest.mark.parametrize(
    ("prepare_jacobian_request", "n_samples"),
    [
        # Only the sample size of the last point is kept,
        # so the Jacobians at an older point
        # are estimated with the current number of samples.
        pytest.param(prepare_older_point, 7, id="older_point"),
        # The statistics of a point sampled by a previous execution
        # are no longer cached,
        # so its sampling for the Jacobians uses the current number of samples.
        pytest.param(prepare_point_of_previous_execution, 7, id="previous_execution"),
        # The last point of a derivative-free execution
        # is sampled again with the size of its statistics,
        # and not with the current number of samples, i.e. 7.
        pytest.param(
            prepare_last_point_after_derivative_free_driver,
            5,
            id="last_point_after_derivative_free_driver",
        ),
    ],
)
def test_sequential_sampling_jacobian_sample_size(
    enable_discipline_statistics: None,
    prepare_jacobian_request: Callable[[], JacobianRequest],
    n_samples: int,
) -> None:
    """Check the number of samples to resample a point for the Jacobians."""
    scenario, compute_jacobian, design_point = prepare_jacobian_request()
    statistics = scenario.disciplines[0].execution_statistics
    n_executions = statistics.n_executions
    compute_jacobian(design_point)
    assert statistics.n_executions - n_executions == n_samples


@pytest.mark.parametrize(
    "during_sampling",
    [
        pytest.param(True, id="during_sampling"),
        # E.g. a driver reusing the array of the last design point for the next one.
        pytest.param(False, id="after_evaluation"),
    ],
)
def test_cache_key_independent_of_input_array(during_sampling: bool) -> None:
    """Check that changing an input array in place does not change the cache key.

    The cache key must be a copy of the input array made before the sampling.
    """
    scenario = create_scenario(Sampling_Settings(n_samples=5), False)
    formulation = scenario.formulation
    objective = formulation.problem.objective.original
    original_compute_samples = formulation.compute_samples
    design_point = design_points[0].copy()

    def change_input_data(problem, input_data, compute_jacobian=False):
        original_compute_samples(problem, input_data, compute_jacobian=compute_jacobian)
        if during_sampling:
            input_data[0] = design_points[1, 0]

    with mock.patch.object(
        formulation, "compute_samples", side_effect=change_input_data
    ):
        objective.evaluate(design_point)

    design_point[0] = design_points[1, 0]
    with mock.patch.object(formulation, "compute_samples") as compute_samples:
        objective.evaluate(design_points[0])

    compute_samples.assert_not_called()


def test_sequential_sampling_key_independent_of_input_array_during_sampling(
    enable_discipline_statistics: None,
) -> None:
    """Check that changing an input array in place during a sampling is harmless.

    The key of the number of samples of the input point
    must be a copy of the input array made before the sampling,
    so that the resampling of this point for the Jacobians
    uses the same number of samples, i.e. 3.
    """
    scenario = create_scenario(create_sequential_sampling_settings(), False)
    formulation = scenario.formulation
    objective = formulation.problem.objective.original
    compute_samples = Sampling.compute_samples

    def change_input_data(self, problem, input_data, compute_jacobian=False):
        compute_samples(self, problem, input_data, compute_jacobian=compute_jacobian)
        input_data[0] = design_points[1, 0]

    with mock.patch.object(
        Sampling, "compute_samples", autospec=True, side_effect=change_input_data
    ):
        objective.evaluate(design_points[0].copy())

    formulation.input_data_to_output_data.clear()
    statistics = scenario.disciplines[0].execution_statistics
    n_executions = statistics.n_executions
    objective.jac(design_points[0])
    assert statistics.n_executions - n_executions == 3


@pytest.fixture
def pce_scenario() -> UMDOScenario:
    """A PCE scenario approximating the Jacobians of the statistics."""
    scenario = create_scenario(
        PCE_Settings(n_samples=5, approximate_statistics_jacobians=True), True
    )
    scenario.formulation.estimate_statistic_jacobians = True
    return scenario


_approximate_name: Final[str] = "_StatisticFunctionForPCE__approximate_jacobians"
"""The name of the method of StatisticFunctionForPCE approximating the Jacobians."""


def test_pce_approximated_jacobians_not_on_evaluation(
    pce_scenario: UMDOScenario,
) -> None:
    """Check that evaluating a statistic does not approximate the Jacobians."""
    with mock.patch.object(
        StatisticFunctionForPCE, _approximate_name, autospec=True
    ) as approximate_jacobians:
        pce_scenario.formulation.problem.objective.evaluate(design_points[0])

    approximate_jacobians.assert_not_called()


def test_pce_approximated_jacobians_on_request(pce_scenario: UMDOScenario) -> None:
    """Check that the approximated Jacobians are computed once for all statistics."""
    problem = pce_scenario.formulation.problem
    design_point = design_points[0]
    problem.objective.evaluate(design_point)
    with mock.patch.object(
        StatisticFunctionForPCE,
        _approximate_name,
        autospec=True,
        side_effect=getattr(StatisticFunctionForPCE, _approximate_name),
    ) as approximate_jacobians:
        problem.objective.jac(design_point)
        for constraint in problem.constraints:
            constraint.jac(design_point)

    approximate_jacobians.assert_called_once()


def test_pce_approximated_jacobians_after_failure(pce_scenario: UMDOScenario) -> None:
    """Check that a failed approximation of the Jacobians can be tried again."""
    objective = pce_scenario.formulation.problem.objective
    design_point = design_points[0]
    objective.evaluate(design_point)
    with (
        mock.patch.object(
            StatisticFunctionForPCE,
            _approximate_name,
            autospec=True,
            side_effect=RuntimeError,
        ),
        pytest.raises(RuntimeError),
    ):
        objective.jac(design_point)

    assert objective.jac(design_point).shape == (1, 1)


@pytest.mark.parametrize(
    ("settings", "statistic_name", "statistic_parameters"),
    [
        pytest.param(
            Sampling_Settings(n_samples=5),
            "Probability",
            {"threshold": 0.0},
            id="Sampling-iterative",
        ),
        pytest.param(
            Sampling_Settings(n_samples=5, estimate_statistics_iteratively=False),
            "Probability",
            {"threshold": 0.0},
            id="Sampling",
        ),
        pytest.param(
            ControlVariate_Settings(n_samples=5), "Mean", {}, id="ControlVariate"
        ),
        pytest.param(Surrogate_Settings(n_samples=5), "Mean", {}, id="Surrogate"),
    ],
)
def test_unsupported_jacobian(
    settings: BaseUMDOFormulationSettings,
    statistic_name: str,
    statistic_parameters: Mapping[str, float],
    snapshot,
) -> None:
    """Check that an unsupported Jacobian raises an error before any sampling.

    The joint sampling of statistics and Jacobians must not be enabled either.
    """
    scenario = create_scenario(settings, False)
    scenario.add_observable("g1", statistic_name, **statistic_parameters)
    formulation = scenario.formulation
    with (
        mock.patch.object(formulation, "compute_samples") as compute_samples,
        assert_exception(NotImplementedError, snapshot),
    ):
        formulation.problem.observables[-1].jac(design_points[0])

    compute_samples.assert_not_called()
    assert not formulation.estimate_statistic_jacobians
