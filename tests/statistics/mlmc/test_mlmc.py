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
"""Tests for the MLMC algorithm."""

from __future__ import annotations

import re
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING
from typing import Callable

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.utils.platform import PLATFORM_IS_WINDOWS
from gemseo.utils.testing.helpers import image_comparison
from numpy import array
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.statistics.multilevel.mlmc.level import Level
from gemseo_umdo.statistics.multilevel.mlmc.mlmc import MLMC
from gemseo_umdo.statistics.multilevel.mlmc.pilots.mean import Mean
from gemseo_umdo.statistics.multilevel.mlmc.pilots.variance import Variance
from gemseo_umdo.use_cases.heat_equation.model import HeatEquationModel
from gemseo_umdo.use_cases.heat_equation.uncertain_space import (
    HeatEquationUncertainSpace,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(scope="module")
def levels() -> list[Level]:
    """The MLMC levels."""
    return [
        Level(lambda x: 2 * x, 2.0),
        Level(lambda x: 1.8 * x, 4.0),
    ]


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace:
    """The uncertain space."""
    space = ParameterSpace()
    space.add_random_variable("x", "OTUniformDistribution")
    return space


@pytest.fixture(scope="module")
def mlmc(levels, uncertain_space):
    """The MLMC algorithm with the default parametrization."""
    return MLMC(levels, uncertain_space, 1000.0)


@pytest.fixture(scope="module")
def executed_mlmc(levels, uncertain_space):
    """The MLMC algorithm with the default parametrization after execution."""
    mlmc = MLMC(levels, uncertain_space, 1000.0)
    mlmc.execute()
    return mlmc


def test_seed_after_instantiation(levels, uncertain_space):
    """Check the seed at the instantiation."""
    assert MLMC(levels, uncertain_space, 1000.0)._MLMC__seed == 0
    assert MLMC(levels, uncertain_space, 1000.0, seed=1)._MLMC__seed == 1


def test_pilot_statistic_estimation_after_instantiation(mlmc):
    """Check that pilot_statistic_estimation is empty after instantiation."""
    assert mlmc.pilot_statistic_estimation.size == 0


def test_sampling_history_after_instantiation(mlmc):
    """Check that sampling_history is empty after instantiation."""
    assert_equal(mlmc.sampling_history, array([[10, 10]]))


def test_budget_history_after_instantiation(mlmc):
    """Check that budget_history is empty after instantiation."""
    assert mlmc.budget_history.size == 0


def test_n_total_samples_after_instantiation(mlmc):
    """Check that n_total_samples is empty after instantiation."""
    assert_equal(mlmc.n_total_samples, array([10, 10]))


def test_model_costs(mlmc):
    """Check model_costs."""
    assert_equal(mlmc.model_costs, array([0.5, 1.0]))


def test_level_costs(mlmc):
    """Check level_costs."""
    assert_equal(mlmc.level_costs, array([0.5, 1.5]))


def test_str(mlmc):
    """Check the string representation of the algorithm."""
    assert str(mlmc) == (
        "Algorithm MLMC\n"
        "   Number of levels: 2\n"
        "   Pilot statistic: Mean\n"
        "   Budget\n"
        "      Minimum: 20.0\n"
        "      Maximum: 1000.0\n"
        "   Numbers of initial samples\n"
        "      n_0 = 10\n"
        "      n_1 = 10\n"
        "   Evaluation costs of the models\n"
        "      C_0 = 0.5\n"
        "      C_1 = 1.0\n"
        "   Evaluation costs of the levels\n"
        "      C_0 = 0.5\n"
        "      C_1 + C_0 = 1.5\n"
        "   Sampling ratios:\n"
        "      r_0 = 2.0\n"
        "      r_1 = 2.0"
    )


def test_maximum_cost_error(levels, uncertain_space):
    """Check that a ValueError is raised if the max cost is lower than the min one."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The minimum budget 20.0 is greater than the total budget 10.0."
        ),
    ):
        MLMC(levels, uncertain_space, 10.0)


def test_pilot_type(mlmc):
    """Check the type of the pilot."""
    assert isinstance(mlmc._MLMC__pilot_statistic_estimator, Mean)


def test_total_n_samples(executed_mlmc):
    """Check the number of samples per level."""
    assert_equal(executed_mlmc.n_total_samples, array([1760, 80]))


def test_budget_history(executed_mlmc):
    """Check the history of the budget."""
    assert_equal(
        executed_mlmc.budget_history,
        array([
            1000.0,
            980.0,
            975.0,
            965.0,
            945.0,
            905.0,
            825.0,
            810.0,
            650.0,
            620.0,
            300.0,
            240.0,
        ]),
    )


def test_sampling_history(executed_mlmc):
    """Check the history of the number of samples per level."""
    assert_equal(
        executed_mlmc.sampling_history,
        array([
            [10, 10],
            [10, 0],
            [20, 0],
            [40, 0],
            [80, 0],
            [160, 0],
            [0, 10],
            [320, 0],
            [0, 20],
            [640, 0],
            [0, 40],
            [480, 0],
        ]),
    )


def test_pilot_statistic_estimation(executed_mlmc):
    """Check the estimation of the pilot statistic."""
    assert_almost_equal(executed_mlmc.pilot_statistic_estimation, array(0.8982755))


def test_pilot(levels, uncertain_space):
    """Check the Variance as pilot, instead of Mean."""
    mlmc = MLMC(levels, uncertain_space, 100.0, pilot_statistic_name="Variance")
    assert isinstance(mlmc._MLMC__pilot_statistic_estimator, Variance)


@pytest.mark.parametrize(
    ("log", "baseline_images"),
    [
        (False, ["mlmc"]),
        (True, ["mlmc_log" if PLATFORM_IS_WINDOWS else "mlmc_log_linux"]),
    ],
)
@image_comparison(None)
def test_plot(baseline_images, executed_mlmc, log):
    """Check the plot of the evaluation history."""
    executed_mlmc.plot_evaluation_history(
        show=False,
        log_n_evaluations=log,
        log_budget=log,
    )


def test_log(levels, uncertain_space, caplog):
    """Check the log of an execution."""
    with (Path(__file__).parent / "mlmc.log").open("r") as f:
        expected = f.read()

    MLMC(levels, uncertain_space, 1000.0).execute()
    log = []
    for record in caplog.record_tuples:
        if record[0] == "gemseo_umdo.statistics.multilevel.mlmc.mlmc":
            assert record[1] == 20
            log.append(record[2])

    assert "\n".join(log) + "\n" == expected


def test_mlmc_without_cost(uncertain_space):
    """Check the use of MLMC without simulation cost provided by the user."""

    def generate_compute_and_wait_function(
        factor: float, secs: float
    ) -> Callable[[NDArray[float]], NDArray[float]]:
        """Generate a function multiplying an input and waiting a few seconds.

        Args:
            factor: The factory of the multiplication.
            secs: The number of seconds to wait.

        Returns:
              The function multiplying an input and waiting a few seconds.
        """

        def compute_and_wait(x: NDArray[float]) -> NDArray[float]:
            """A function multiplying an input and waiting a few seconds.

            Args:
                x: The input.

            Returns:
                The output.
            """
            y = factor * x
            sleep(secs)
            return y

        return compute_and_wait

    levels = [
        Level(generate_compute_and_wait_function(2, 1)),
        Level(generate_compute_and_wait_function(1.8, 1.1)),
    ]
    mlmc = MLMC(levels, uncertain_space, 50.0)
    mlmc.execute()
    assert 0 < mlmc.n_total_samples[1] < mlmc.n_total_samples[0]


def test_stop_when_sampling_is_too_expensive(caplog):
    """Check that the algorithm stops when sampling l_star is too expensive."""
    mesh_sizes = [15, 30, 60, 120]
    disciplines = [HeatEquationModel(mesh_size) for mesh_size in mesh_sizes]

    class Model:
        def __init__(self, discipline):
            self.discipline = discipline

        def __call__(self, x):
            return self.discipline(x)[0][:, None]

    levels = [
        Level(
            Model(discipline),
            cost=discipline.configuration.cost,
            sampling_ratio=1.1,
            n_initial_samples=10,
        )
        for discipline in disciplines
    ]

    mlmc = MLMC(levels, HeatEquationUncertainSpace(), 100)
    mlmc.execute()
    assert caplog.record_tuples[822] == (
        "gemseo_umdo.statistics.multilevel.mlmc.mlmc",
        20,
        "Stop the algorithm as sampling l_star is too expensive.",
    )
    assert mlmc.pilot_statistic_estimation == pytest.approx(37.82445701860196)
