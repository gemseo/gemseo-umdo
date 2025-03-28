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

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo.algos.parameter_space import ParameterSpace
from numpy import array
from pandas._testing import assert_frame_equal

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.problems.uopt_as_umdo_scenario import UOptAsUMDOScenario
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario


def test_u_opt_as_umdo_scenario():
    """Check that the UOptAsUMDOScenario gives the same results as its monodisciplinary counterpart."""  # noqa: E501
    discipline = create_discipline(
        "AnalyticDiscipline",
        expressions={
            "f": "100*(z_2-(u*z_1)**2)**2+(1-v*z_1)**2+100*(z_1-(u*z_0)**2)**2+(1-v*z_0)**2"  # noqa: E501
        },
        name="Rosenbrock",
    )

    design_space = create_design_space()
    design_space.add_variable("z_0", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_1", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_2", lower_bound=-1, upper_bound=1)

    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("u", "OTNormalDistribution", mu=1.0, sigma=0.01)
    uncertain_space.add_random_variable("v", "OTNormalDistribution", mu=1.0, sigma=0.01)

    initial_point = array([-0.25, 0.75, -0.9])
    design_space.set_current_value(initial_point)

    u_opt_scenario = UMDOScenario(
        [discipline],
        "f",
        design_space,
        uncertain_space,
        "Mean",
        Sampling_Settings(n_samples=5, estimate_statistics_iteratively=False),
        formulation_name="DisciplinaryOpt",
    )
    u_opt_scenario.execute(algo_name="NLOPT_SLSQP", max_iter=5)

    design_space.set_current_value(initial_point)

    umdo_scenario = UOptAsUMDOScenario(
        discipline,
        "f",
        design_space,
        uncertain_space,
        "Mean",
        Sampling_Settings(n_samples=5, estimate_statistics_iteratively=False),
        formulation_name="MDF",
    )
    umdo_scenario.execute(algo_name="NLOPT_SLSQP", max_iter=5)

    assert_frame_equal(
        umdo_scenario.formulation.optimization_problem.database.to_dataset(),
        u_opt_scenario.formulation.optimization_problem.database.to_dataset(),
        atol=1e-5,
    )
