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

from typing import TYPE_CHECKING

import pytest
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_parameter_space
from gemseo.problems.mdo.opt_as_mdo_scenario import LinearLinkDiscipline
from gemseo.problems.mdo.opt_as_mdo_scenario import create_disciplines
from gemseo.scenarios.mdo_scenario import MDOScenario
from numpy import array
from openturns.testing import assert_almost_equal

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.typing import RealArray


@pytest.fixture
def design_space() -> DesignSpace:
    """The design space."""
    ds = create_design_space()
    ds.add_variable("z_0", lower_bound=-1, upper_bound=1)
    ds.add_variable("z_1", lower_bound=-1, upper_bound=1)
    ds.add_variable("z_2", lower_bound=-1, upper_bound=1)
    return ds


@pytest.fixture
def rosenbrock() -> AnalyticDiscipline:
    """The Rosenbrock discipline."""
    discipline = create_discipline(
        "AnalyticDiscipline",
        expressions={
            "f": "100*(z_2-(u*z_1)**2)**2+(1-v*z_1)**2+100*(z_1-(u*z_0)**2)**2+(1-v*z_0)**2"  # noqa: E501
        },
        name="Rosenbrock",
    )
    discipline.io.input_grammar.defaults.update({"u": array([1.0]), "v": array([1.0])})
    return discipline


@pytest.fixture
def disciplines(
    design_space, rosenbrock
) -> tuple[Discipline, LinearLinkDiscipline, Discipline]:
    """The disciplines of the multidisciplinary Rosenbrock problem."""
    return create_disciplines(rosenbrock, design_space, (), LinearLinkDiscipline)


@pytest.fixture
def sub_scenarios(design_space, disciplines):
    """The sub-scenarios used by the BiLevel formulation."""
    scenario_1 = MDOScenario(
        [disciplines[2], disciplines[1], disciplines[0]],
        "f",
        design_space.filter("x_1", copy=True),
        formulation_name="DisciplinaryOpt",
    )
    scenario_1.set_algorithm(algo_name="SLSQP", max_iter=10)

    scenario_2 = MDOScenario(
        [disciplines[3], disciplines[1], disciplines[0]],
        "f",
        design_space.filter("x_2", copy=True),
        formulation_name="DisciplinaryOpt",
    )
    scenario_2.set_algorithm(algo_name="SLSQP", max_iter=10)

    return scenario_1, scenario_2


@pytest.fixture
def reference_database(
    design_space, rosenbrock, sub_scenarios
) -> tuple[RealArray, RealArray]:
    """The reference database obtained using a BiLevel-based MDOScenario."""
    bilevel_scenario = MDOScenario(
        [*sub_scenarios, rosenbrock],
        "f",
        design_space.filter("x_0", copy=True),
        formulation_name="BiLevel",
    )
    bilevel_scenario.execute(algo_name="CustomDOE", samples=array([[1.0]]))
    database = bilevel_scenario.formulation.optimization_problem.database
    return database.get_function_history("f", with_x_vect=True)


def test_u_bilevel(design_space, rosenbrock, sub_scenarios, reference_database):
    """Verify that UMDOScenario supports the BiLevel formulation."""
    uncertain_space = create_parameter_space()
    uncertain_space.add_random_variable("u", "OTDiracDistribution", variable_value=1.0)
    uncertain_space.add_random_variable("v", "OTDiracDistribution", variable_value=1.0)

    u_bilevel_scenario = UMDOScenario(
        [*sub_scenarios, rosenbrock],
        "f",
        design_space.filter("x_0", copy=True),
        uncertain_space,
        "Mean",
        Sampling_Settings(n_samples=3, estimate_statistics_iteratively=False),
        formulation_name="BiLevel",
    )
    u_bilevel_scenario.execute(algo_name="CustomDOE", samples=array([[1.0]]))

    database = u_bilevel_scenario.formulation.optimization_problem.database
    f_history, x_0_history = database.get_function_history("E[f]", with_x_vect=True)

    assert_almost_equal(f_history, reference_database[0])
    assert_almost_equal(x_0_history, reference_database[1])
