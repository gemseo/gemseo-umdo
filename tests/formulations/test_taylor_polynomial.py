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
from gemseo import from_pickle
from gemseo import to_pickle
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.mdf import MDF
from numpy import array
from numpy import ndarray
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.formulations._statistics.taylor_polynomial.margin import Margin
from gemseo_umdo.formulations._statistics.taylor_polynomial.mean import Mean
from gemseo_umdo.formulations._statistics.taylor_polynomial.standard_deviation import (
    StandardDeviation,
)
from gemseo_umdo.formulations._statistics.taylor_polynomial.variance import Variance
from gemseo_umdo.formulations.taylor_polynomial import TaylorPolynomial
from gemseo_umdo.formulations.taylor_polynomial_settings import (
    TaylorPolynomial_Settings,
)
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline.discipline import Discipline


@pytest.fixture
def umdo_formulation(
    disciplines: Sequence[Discipline],
    design_space: DesignSpace,
    mdo_formulation: MDF,
    uncertain_space: ParameterSpace,
) -> TaylorPolynomial:
    """The UMDO formulation based on Taylor polynomial."""
    design_space = MDF(disciplines, "f", design_space).design_space
    formulation = TaylorPolynomial(
        disciplines,
        "f",
        design_space,
        mdo_formulation,
        uncertain_space,
        "Mean",
        TaylorPolynomial_Settings(),
    )
    formulation.add_constraint("c", "Mean")
    formulation.add_observable("o", "Mean")
    return formulation


@pytest.fixture
def umdo_formulation_with_hessian(
    disciplines: Sequence[Discipline],
    design_space: DesignSpace,
    mdo_formulation: MDF,
    uncertain_space: ParameterSpace,
) -> TaylorPolynomial:
    """The UMDO formulation based on second-order approximation."""
    design_space = MDF(disciplines, "f", design_space).design_space
    formulation = TaylorPolynomial(
        disciplines,
        "f",
        design_space,
        mdo_formulation,
        uncertain_space,
        "Mean",
        TaylorPolynomial_Settings(second_order=True),
    )
    formulation.add_constraint("c", "Mean")
    formulation.add_observable("o", "Mean")
    return formulation


@pytest.fixture
def scenario_input_data() -> dict[str, str | dict[str, ndarray]]:
    """The input data of the scenario."""
    return {
        "algo_name": "CustomDOE",
        "samples": array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    }


@pytest.fixture(params=[False, True])
def scenario(disciplines, design_space, uncertain_space, request):
    """A scenario of interest."""
    scn = UDOEScenario(
        disciplines,
        "f",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="MDF",
        statistic_estimation_settings=TaylorPolynomial_Settings(
            second_order=request.param
        ),
    )
    scn.add_constraint("c", "Margin", factor=3.0)
    scn.add_observable("o", "Variance")
    return scn


def test_scenario_execution(scenario, scenario_input_data):
    """Check the execution of an UMDOScenario with the TaylorPolynomial formulation."""
    scenario.execute(**scenario_input_data)
    optimization_result = scenario.optimization_result
    assert_equal(optimization_result.x_opt, array([1.0, 1.0, 1.0]))
    assert_almost_equal(optimization_result.f_opt, array([-21.0]))


def test_scenario_serialization(scenario, tmp_path, scenario_input_data):
    """Check the serialization of an UMDOScenario with Sampling U-MDO formulation."""
    file_path = tmp_path / "scenario.h5"
    to_pickle(scenario, file_path)
    saved_scenario = from_pickle(file_path)
    saved_scenario.execute(**scenario_input_data)
    optimization_result = saved_scenario.optimization_result
    assert_equal(optimization_result.x_opt, array([1.0, 1.0, 1.0]))
    assert_almost_equal(optimization_result.f_opt, array([-21.0]))


# In the following, we will name m and s the mean and standard deviation of U.
# Here are the value of f(x, m), jac(x, m) and hess(x, m).
FUNC = array([1.0, 2.0])
JAC = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
HESS = array([
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]],
])


def test_init_differentiation_method(umdo_formulation):
    """Check that the default method to derive wrt uncertainties is USER_GRAD."""
    problem = umdo_formulation.mdo_formulation.optimization_problem
    assert problem.differentiation_method == problem.DifferentiationMethod.USER_GRAD


def test_estimate_mean(umdo_formulation):
    """Check that E[f(x,U)] is estimated by f(x,m) + 0.5s²h(x,m)."""
    mean = Mean(umdo_formulation.uncertain_space)
    mean_estimation = mean.estimate_statistic(FUNC, JAC, HESS)
    assert_equal(mean_estimation, array([115.0, -112.0]))


def test_estimate_variance(umdo_formulation):
    """Check that V[f(x,U)] is estimated by (sj(x,m))²."""
    variance = Variance(umdo_formulation.uncertain_space)
    var_estimation = variance.estimate_statistic(FUNC, JAC, HESS)
    assert_equal(var_estimation, array([98.0, 440.0]))


def test_estimate_standard_derivation(umdo_formulation):
    """Check that V[f(x,u)] = S[f(x,U)]²."""
    standard_deviation = StandardDeviation(umdo_formulation.uncertain_space)
    std_estimation = standard_deviation.estimate_statistic(FUNC, JAC, HESS)
    assert_equal(std_estimation, array([98.0, 440.0]) ** 0.5)


def test_estimate_margin(umdo_formulation):
    """Check that Margin[f(x,u);k] = E[f(x,U)] + kS[f(x,U)]."""
    margin = Margin(umdo_formulation.uncertain_space, factor=3.0)
    margin_estimation = margin.estimate_statistic(FUNC, JAC, HESS)
    assert_equal(
        margin_estimation, array([115.0, -112.0]) + 3.0 * array([98.0, 440.0]) ** 0.5
    )


def test_mdo_formulation_objective(umdo_formulation, mdf_discipline):
    """Check that the MDO formulation can compute the objective correctly."""
    objective = umdo_formulation.mdo_formulation.optimization_problem.objective
    input_data = {name: array([2.0]) for name in ["u", "u1", "u2"]}
    assert_equal(
        objective.evaluate(array([2.0] * 3)), mdf_discipline.execute(input_data)["f"]
    )


def test_mdo_formulation_constraint(umdo_formulation, mdf_discipline):
    """Check that the MDO formulation can compute the constraints correctly."""
    constraint = umdo_formulation.mdo_formulation.optimization_problem.observables[0]
    input_data = {name: array([2.0]) for name in ["u", "u1", "u2"]}
    assert_equal(
        constraint.evaluate(array([2.0] * 3)), mdf_discipline.execute(input_data)["c"]
    )


def test_mdo_formulation_observable(umdo_formulation, mdf_discipline):
    """Check that the MDO formulation can compute the observables correctly."""
    observable = umdo_formulation.mdo_formulation.optimization_problem.observables[1]
    input_data = {name: array([2.0]) for name in ["u", "u1", "u2"]}
    assert_equal(
        observable.evaluate(array([2.0] * 3)), mdf_discipline.execute(input_data)["o"]
    )


def test_umdo_formulation_objective(umdo_formulation, mdf_discipline):
    """Check that the UMDO formulation can compute the objective correctly."""
    objective = umdo_formulation.optimization_problem.objective
    uncertain_space = umdo_formulation.uncertain_space
    input_data = uncertain_space.convert_array_to_dict(
        uncertain_space.distribution.mean
    )
    assert_almost_equal(
        objective.evaluate(array([0.0] * 3)), mdf_discipline.execute(input_data)["f"]
    )


def test_umdo_formulation_constraint(umdo_formulation, mdf_discipline):
    """Check that the UMDO formulation can compute the constraints correctly."""
    constraint = umdo_formulation.optimization_problem.constraints[0]
    uncertain_space = umdo_formulation.uncertain_space
    input_data = uncertain_space.convert_array_to_dict(
        uncertain_space.distribution.mean
    )
    assert_almost_equal(
        constraint.evaluate(array([0.0] * 3)), mdf_discipline.execute(input_data)["c"]
    )


def test_umdo_formulation_observable(umdo_formulation, mdf_discipline):
    """Check that the UMDO formulation can compute the observables correctly."""
    observable = umdo_formulation.optimization_problem.observables[0]
    uncertain_space = umdo_formulation.uncertain_space
    input_data = uncertain_space.convert_array_to_dict(
        uncertain_space.distribution.mean
    )
    assert_almost_equal(
        observable.evaluate(array([0.0] * 3)), mdf_discipline.execute(input_data)["o"]
    )


def test_second_order_approximation(umdo_formulation_with_hessian):
    """Check second-order approximation."""
    problem = umdo_formulation_with_hessian.hessian_fd_problem
    objective = problem.objective
    assert objective.name == "@@f"
    objective_value = objective.evaluate(array([0.0] * 3))
    assert objective_value.shape == (3, 3)
    assert_equal(objective_value, 0.0)

    constraint = problem.observables[0]
    assert constraint.name == "@@c"
    constraint_value = constraint.evaluate(array([0.0] * 3))
    assert constraint_value.shape == (3, 3)
    assert_equal(constraint_value, 0.0)


def test_uncertain_input_data_non_normalization():
    """Check that the uncertain input data non-normalization is managed correcty."""
    discipline = AnalyticDiscipline({"f": "x+u"})
    design_space = DesignSpace()
    design_space.add_variable("x")
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable(
        "u", "OTUniformDistribution", minimum=0.0, maximum=1.5
    )
    scenario = UDOEScenario(
        [discipline],
        "f",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation_settings=TaylorPolynomial_Settings(),
        formulation_name="DisciplinaryOpt",
    )
    scenario.execute(CustomDOE_Settings(samples=array([[1.0]]), eval_jac=True))
    assert_almost_equal(discipline.io.data["x"], array([1.0]))
    # u = 1.125, f = 1+1.125² and dfdu = 2.25 before bug fix
    assert_almost_equal(discipline.io.data["u"], array([0.75]))
    assert_almost_equal(discipline.io.data["f"], array([1.75]))
    assert_almost_equal(discipline.jac["f"]["u"], array([[1.0]]))
