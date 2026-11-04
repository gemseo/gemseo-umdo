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

import logging

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import OT_MONTE_CARLO_Settings
from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.doe import PYDOE_FULLFACT_Settings
from gemseo.optimization import OptimizationProblem
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.util.testing.helper import assert_exception
from numpy import array
from numpy.testing import assert_almost_equal

from gemseo_umdo.formulations._functions import statistic_function_for_sampling
from gemseo_umdo.formulations.sequential_sampling import SequentialSampling
from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario


def increment(n_samples: int) -> int:
    """Compute the increment of the sampling size.

    Args:
        n_samples: The current number of samples.

    Returns:
        The increment of the sampling size.
    """
    return 2


@pytest.fixture
def discipline() -> AnalyticDiscipline:
    """The discipline computing y=(x+u)**2."""
    return AnalyticDiscipline({"y": "(x+u)**2"}, name="quadratic_function")


@pytest.fixture
def design_space() -> DesignSpace:
    """The design space, with x in [-1, 1]."""
    space = DesignSpace()
    space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)
    return space


@pytest.fixture
def uncertain_space() -> RandomSpace:
    """The uncertain space, with u normally distributed."""
    space = RandomSpace()
    space.add_variable("u", OTNormalDistribution_Settings())
    return space


@pytest.mark.parametrize("add_constraint", [False, True])
@pytest.mark.parametrize("estimate_statistics_iteratively", [False, True])
@pytest.mark.parametrize("n_samples_increment", [2, increment])
def test_scenario(
    discipline,
    design_space,
    uncertain_space,
    estimate_statistics_iteratively,
    enable_discipline_statistics,
    n_samples_increment,
    add_constraint,
):
    """Check SequentialSampling.

    A constraint must not increase the number of samples.
    """
    scenario = UMDOScenario(
        [discipline],
        design_space,
        uncertain_space,
        statistic_estimation_settings=SequentialSampling_Settings(
            doe_algo_settings=OT_OPT_LHS_Settings(n_samples=7),
            initial_n_samples=3,
            n_samples_increment=n_samples_increment,
            estimate_statistics_iteratively=estimate_statistics_iteratively,
        ),
    )
    scenario.add_objective("y", "Mean")
    if add_constraint:
        scenario.add_constraint("y", "Margin")
    scenario.execute(PYDOE_FULLFACT_Settings(n_samples=5))
    assert discipline.execution_statistics.n_executions == (3 + 5 + 7 + 7 + 7)


def test_warning_with_jacobian_resampling(
    discipline, design_space, uncertain_space, monkeypatch, caplog
):
    """Check the warning when the Jacobian is estimated from a new sampling.

    The estimation of the statistics uses 2 samples, all complete,
    and that of their Jacobians uses a new sampling with the same number of samples,
    one without gradient;
    this failure is reported at the warning level.
    """
    caplog.set_level(logging.INFO)
    compute_jacobian = discipline._compute_jacobian
    n_calls = []

    def _compute_jacobian(*args, **kwargs):
        n_calls.append(None)
        if len(n_calls) == 1:
            msg = "Failure of _compute_jacobian."
            raise ValueError(msg)
        return compute_jacobian(*args, **kwargs)

    monkeypatch.setattr(discipline, "_compute_jacobian", _compute_jacobian)
    formulation = SequentialSampling(
        OptimizationProblem(design_space),
        [discipline],
        SequentialSampling_Settings(
            doe_algo_settings=OT_MONTE_CARLO_Settings(n_samples=3),
            initial_n_samples=2,
            estimate_statistics_iteratively=False,
        ),
        uncertain_space=uncertain_space,
    )
    function = formulation.problem.objective = formulation.create_objective("y", "Mean")
    function.evaluate(array([0.5]))
    function.jac(array([0.5]))
    assert [
        (level, text)
        for name, level, text in caplog.record_tuples
        if name == statistic_function_for_sampling.__name__
    ] == [
        (
            logging.WARNING,
            (
                "Some outputs or gradients are missing at some of the 2 samples "
                "of the uncertain variables, e.g. because their evaluation failed; "
                "the numbers of samples used to estimate the statistics are: "
                "1 for y and its gradient."
            ),
        )
    ]


@pytest.fixture(
    params=[pytest.param(False, id="standard"), pytest.param(True, id="iterative")]
)
def formulation_with_one_initial_sample(
    request, discipline, design_space, uncertain_space
) -> SequentialSampling:
    """A SequentialSampling starting from a single sample.

    It estimates the statistics iteratively or not.
    """
    return SequentialSampling(
        OptimizationProblem(design_space),
        [discipline],
        SequentialSampling_Settings(
            doe_algo_settings=OT_MONTE_CARLO_Settings(n_samples=3),
            initial_n_samples=1,
            estimate_statistics_iteratively=request.param,
        ),
        uncertain_space=uncertain_space,
    )


def test_initial_n_samples_one_mean(formulation_with_one_initial_sample):
    """Check that a mean can be estimated from a single initial sample."""
    formulation = formulation_with_one_initial_sample
    function = formulation.create_objective("y", "Mean")
    mean = function.evaluate(array([0.5]))
    ((u,),) = formulation.doe_algo.samples
    assert_almost_equal(mean, array([(0.5 + u) ** 2]))


def test_initial_n_samples_one_variance(formulation_with_one_initial_sample, snapshot):
    """Check that a variance cannot be estimated from a single initial sample."""
    function = formulation_with_one_initial_sample.create_objective("y", "Variance")
    with assert_exception(ValueError, snapshot):
        function.evaluate(array([0.5]))
