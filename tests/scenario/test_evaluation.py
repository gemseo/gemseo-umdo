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
from gemseo.doe import CustomDOE_Settings
from gemseo.optimization import OptimizationProblem
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from numpy import array
from numpy.testing import assert_almost_equal

from gemseo_umdo.formulation.sampling import Sampling
from gemseo_umdo.formulation.sampling_settings import Sampling_Settings
from gemseo_umdo.scenario.evaluation import UEvaluationScenario
from gemseo_umdo.statistic.settings.margin import Margin_Settings


@pytest.fixture
def u_evaluation_scenario(
    disciplines, design_space, uncertain_space
) -> UEvaluationScenario:
    """A UEvaluationScenario with an uncertain design variable."""
    return UEvaluationScenario(
        disciplines,
        design_space,
        uncertain_space,
        Sampling_Settings(n_samples=3),
        uncertain_design_variables={"x0": ("+", "u")},
    )


@pytest.mark.parametrize("method_name", ["add_objective", "add_constraint"])
def test_u_evaluation_scenario_no_objective_nor_constraint(
    u_evaluation_scenario, method_name
):
    """Check that a UEvaluationScenario cannot define an objective or constraints."""
    assert not hasattr(u_evaluation_scenario, method_name)


def test_u_evaluation_scenario_problem(u_evaluation_scenario):
    """Check that a UEvaluationScenario does not define an optimization problem."""
    assert not isinstance(
        u_evaluation_scenario.formulation.problem, OptimizationProblem
    )


def test_u_evaluation_scenario_formulation(u_evaluation_scenario):
    """Check that a UEvaluationScenario uses the U-MDO formulation of its settings."""
    assert isinstance(u_evaluation_scenario.formulation, Sampling)


def test_u_evaluation_scenario_repr(u_evaluation_scenario):
    """Check the string representation of a UEvaluationScenario."""
    assert repr(u_evaluation_scenario).startswith("UEvaluationScenario\n")


def test_u_evaluation_scenario_input_space(u_evaluation_scenario):
    """Check that the input space of a UEvaluationScenario includes the noising."""
    assert set(u_evaluation_scenario.input_space.variables) == {"dv_x0", "x1", "x2"}


def test_u_evaluation_scenario_observables(u_evaluation_scenario):
    """Check that a UEvaluationScenario evaluates statistics as observables."""
    u_evaluation_scenario.add_observable("o", "Mean")
    u_evaluation_scenario.add_observable("o", Margin_Settings(factor=2.0))
    u_evaluation_scenario.execute(
        CustomDOE_Settings(samples=array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    )
    dataset = u_evaluation_scenario.to_dataset()
    assert set(dataset.output_names) == {"E[o]", "Margin[o; 2.0]"}
    assert len(dataset) == 2


def test_u_evaluation_scenario_values():
    """Check the statistics estimated by a UEvaluationScenario.

    The mean of $(x+u)^2$ over the samples $u=-1$ and $u=1$
    is the expectation $x^2+1$ of $(x+U)^2$ where $U$ is a standard Gaussian variable.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1.0, upper_bound=1.0)

    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", OTNormalDistribution_Settings())

    scenario = UEvaluationScenario(
        [AnalyticDiscipline({"y": "(x+u)**2"})],
        design_space,
        uncertain_space,
        Sampling_Settings(
            doe_algo_settings=CustomDOE_Settings(samples=array([[-1.0], [1.0]]))
        ),
    )
    scenario.add_observable("y", "Mean")
    scenario.execute(CustomDOE_Settings(samples=array([[-0.5], [0.0], [0.5]])))
    mean = scenario.to_dataset().get_view(variable_names="E[y]").to_numpy()
    assert_almost_equal(mean, array([[1.25], [1.0], [1.25]]))
