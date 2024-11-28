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
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline

from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario


@pytest.mark.parametrize("estimate_statistics_iteratively", [False, True])
def test_scenario(estimate_statistics_iteratively):
    """Check SequentialSampling."""
    discipline = AnalyticDiscipline({"y": "(x+u)**2"}, name="quadratic_function")
    discipline.set_cache("MemoryFullCache")
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("u", "OTNormalDistribution")
    scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="DisciplinaryOpt",
        statistic_estimation_settings=SequentialSampling_Settings(
            doe_algo_settings=OT_OPT_LHS_Settings(n_samples=7),
            initial_n_samples=3,
            n_samples_increment=2,
            estimate_statistics_iteratively=estimate_statistics_iteratively,
        ),
    )
    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=5)
    assert discipline.execution_statistics.n_executions == (3 + 5 + 7 + 7 + 7)
