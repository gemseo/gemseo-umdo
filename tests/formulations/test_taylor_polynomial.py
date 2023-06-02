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

from typing import Sequence

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.formulations.mdf import MDF
from gemseo_umdo.estimators.taylor_polynomial import Margin
from gemseo_umdo.estimators.taylor_polynomial import Mean
from gemseo_umdo.estimators.taylor_polynomial import StandardDeviation
from gemseo_umdo.estimators.taylor_polynomial import Variance
from gemseo_umdo.formulations.taylor_polynomial import TaylorPolynomial
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario
from numpy import array
from numpy.testing import assert_equal


@pytest.fixture
def umdo_formulation(
    disciplines: Sequence[MDODiscipline],
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
    )
    formulation.add_constraint("c", "Mean")
    formulation.add_observable("o", "Mean")
    return formulation


@pytest.fixture
def umdo_formulation_with_hessian(
    disciplines: Sequence[MDODiscipline],
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
        second_order=True,
    )
    formulation.add_constraint("c", "Mean")
    formulation.add_observable("o", "Mean")
    return formulation


@pytest.mark.parametrize("second_order", [False, True])
def test_scenario(
    disciplines, design_space, uncertain_space, second_order, mdf_discipline, tmp_path
):
    """Check the optimum returned by the UMDOScenario."""
    scn = UDOEScenario(
        disciplines,
        "MDF",
        "f",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation="TaylorPolynomial",
        statistic_estimation_parameters={"second_order": second_order},
    )
    scn.add_constraint("c", "Margin", factor=3.0)
    scn.add_observable("o", "Variance")
    file_path = tmp_path / "scenario.h5"
    scn.to_pickle(file_path)
    algo_data = {"algo": "CustomDOE", "algo_options": {"samples": array([[0.0] * 3])}}
    scn.execute(algo_data)
    saved_scn = UDOEScenario.from_pickle(file_path)
    saved_scn.execute(algo_data)
    uncertain_data = uncertain_space.array_to_dict(uncertain_space.distribution.mean)
    expected_output_data = mdf_discipline.execute(uncertain_data)["f"]
    expected_input_data = array([0.0] * 3)
    assert_equal(scn.optimization_result.x_opt, expected_input_data)
    assert_equal(scn.optimization_result.f_opt, expected_output_data)
    assert_equal(saved_scn.optimization_result.x_opt, expected_input_data)
    assert_equal(saved_scn.optimization_result.f_opt, expected_output_data)


# In the following, we will name m and s the mean and standard deviation of U.
# Here are the value of f(x, m), jac(x, m) and hess(x, m).
FUNC = array([1.0, 2.0])
JAC = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
HESS = array(
    [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]],
    ]
)


def test_init_differentiation_method(umdo_formulation):
    """Check that the default method to derive wrt uncertainties is USER_GRAD."""
    problem = umdo_formulation.mdo_formulation.opt_problem
    assert problem.differentiation_method == problem.DifferentiationMethod.USER_GRAD


def test_estimate_mean(umdo_formulation):
    """Check that E[f(x,U)] is estimated by f(x,m) + 0.5s²h(x,m)."""
    mean_estimation = Mean(umdo_formulation)(FUNC, JAC, HESS)
    assert_equal(mean_estimation, array([115.0, -112.0]))


def test_estimate_variance(umdo_formulation):
    """Check that V[f(x,U)] is estimated by (sj(x,m))²."""
    var_estimation = Variance(umdo_formulation)(FUNC, JAC, HESS)
    assert_equal(var_estimation, array([98.0, 440.0]))


def test_estimate_standard_derivation(umdo_formulation):
    """Check that V[f(x,u)] = S[f(x,U)]²."""
    std_estimation = StandardDeviation(umdo_formulation)(FUNC, JAC, HESS)
    assert_equal(std_estimation, array([98.0, 440.0]) ** 0.5)


def test_estimate_margin(umdo_formulation):
    """Check that Margin[f(x,u);k] = E[f(x,U)] + kS[f(x,U)]."""
    margin_estimation = Margin(umdo_formulation)(FUNC, JAC, hess=HESS, factor=3.0)
    assert_equal(
        margin_estimation, array([115.0, -112.0]) + 3.0 * array([98.0, 440.0]) ** 0.5
    )


def test_mdo_formulation_objective(umdo_formulation, mdf_discipline):
    """Check that the MDO formulation can compute the objective correctly."""
    objective = umdo_formulation.mdo_formulation.opt_problem.objective
    input_data = {name: array([2.0]) for name in ["u", "u1", "u2"]}
    assert_equal(objective(array([2.0] * 3)), mdf_discipline.execute(input_data)["f"])


def test_mdo_formulation_constraint(umdo_formulation, mdf_discipline):
    """Check that the MDO formulation can compute the constraints correctly."""
    constraint = umdo_formulation.mdo_formulation.opt_problem.constraints[0]
    input_data = {name: array([2.0]) for name in ["u", "u1", "u2"]}
    assert_equal(constraint(array([2.0] * 3)), mdf_discipline.execute(input_data)["c"])


def test_mdo_formulation_observable(umdo_formulation, mdf_discipline):
    """Check that the MDO formulation can compute the observables correctly."""
    observable = umdo_formulation.mdo_formulation.opt_problem.observables[0]
    input_data = {name: array([2.0]) for name in ["u", "u1", "u2"]}
    assert_equal(observable(array([2.0] * 3)), mdf_discipline.execute(input_data)["o"])


def test_umdo_formulation_objective(umdo_formulation, mdf_discipline):
    """Check that the UMDO formulation can compute the objective correctly."""
    objective = umdo_formulation.opt_problem.objective
    uncertain_space = umdo_formulation.uncertain_space
    input_data = uncertain_space.array_to_dict(uncertain_space.distribution.mean)
    assert_equal(objective(array([0.0] * 3)), mdf_discipline.execute(input_data)["f"])


def test_umdo_formulation_constraint(umdo_formulation, mdf_discipline):
    """Check that the UMDO formulation can compute the constraints correctly."""
    constraint = umdo_formulation.opt_problem.constraints[0]
    uncertain_space = umdo_formulation.uncertain_space
    input_data = uncertain_space.array_to_dict(uncertain_space.distribution.mean)
    assert_equal(constraint(array([0.0] * 3)), mdf_discipline.execute(input_data)["c"])


def test_umdo_formulation_observable(umdo_formulation, mdf_discipline):
    """Check that the UMDO formulation can compute the observables correctly."""
    observable = umdo_formulation.opt_problem.observables[0]
    uncertain_space = umdo_formulation.uncertain_space
    input_data = uncertain_space.array_to_dict(uncertain_space.distribution.mean)
    assert_equal(observable(array([0.0] * 3)), mdf_discipline.execute(input_data)["o"])


def test_second_order_approximation(umdo_formulation_with_hessian):
    """Check second-order approximation."""
    problem = umdo_formulation_with_hessian.hessian_fd_problem
    objective = problem.objective
    assert objective.name == "@@f"
    objective_value = objective(array([0.0] * 3))
    assert objective_value.shape == (3, 3)
    assert_equal(objective_value, 0.0)

    constraint = problem.constraints[0]
    assert constraint.name == "@@c"
    constraint_value = constraint(array([0.0] * 3))
    assert constraint_value.shape == (3, 3)
    assert_equal(constraint_value, 0.0)
