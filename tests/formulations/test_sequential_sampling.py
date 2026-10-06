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
from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.doe import PYDOE_FULLFACT_Settings
from gemseo.formulation import DisciplinaryOpt_Settings
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings

from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario


def increment(n_samples: int) -> int:
    """Compute the increment of the sampling size.

    Args:
        n_samples: The current number of samples.

    Returns:
        The increment of the sampling size.
    """
    return 2


@pytest.mark.parametrize("estimate_statistics_iteratively", [False, True])
@pytest.mark.parametrize("n_samples_increment", [2, increment])
def test_scenario(
    estimate_statistics_iteratively, enable_discipline_statistics, n_samples_increment
):
    """Check SequentialSampling."""
    discipline = AnalyticDiscipline({"y": "(x+u)**2"}, name="quadratic_function")
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)
    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", OTNormalDistribution_Settings())
    scenario = UDOEScenario(
        [discipline],
        design_space,
        uncertain_space,
        statistic_estimation_settings=SequentialSampling_Settings(
            doe_algo_settings=OT_OPT_LHS_Settings(n_samples=7),
            initial_n_samples=3,
            n_samples_increment=n_samples_increment,
            estimate_statistics_iteratively=estimate_statistics_iteratively,
        ),
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    scenario.add_objective("y", "Mean")
    scenario.execute(algorithm_settings=PYDOE_FULLFACT_Settings(n_samples=5))
    assert discipline.execution_statistics.n_executions == (3 + 5 + 7 + 7 + 7)
