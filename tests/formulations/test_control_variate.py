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
"""Test for the UMDO formulation ControlVariate."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest
from gemseo.formulations.mdf import MDF
from numpy import array
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.formulations._statistics.control_variate.margin import Margin
from gemseo_umdo.formulations._statistics.control_variate.mean import Mean
from gemseo_umdo.formulations._statistics.control_variate.probability import Probability
from gemseo_umdo.formulations._statistics.control_variate.standard_deviation import (
    StandardDeviation,
)
from gemseo_umdo.formulations._statistics.control_variate.variance import Variance
from gemseo_umdo.formulations.control_variate import ControlVariate
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline


@pytest.fixture
def umdo_formulation(
    disciplines: Sequence[MDODiscipline],
    design_space: DesignSpace,
    mdo_formulation: MDF,
    uncertain_space: ParameterSpace,
) -> ControlVariate:
    """The UMDO formulation."""
    design_space = MDF(disciplines, "f", design_space).design_space
    formulation = ControlVariate(
        disciplines,
        "f",
        design_space,
        mdo_formulation,
        uncertain_space,
        "Mean",
        n_samples=None,
        algo="CustomDOE",
        algo_options={"samples": array([[0.0] * 3, [1.0] * 3])},
    )
    formulation.add_constraint("c", "Mean")
    formulation.add_observable("o", "Mean")
    return formulation


@pytest.fixture(scope="module")
def algo_data() -> dict[str, Any]:
    """Input data for a DOE-based u-scenario."""
    return {"algo": "CustomDOE", "algo_options": {"samples": array([[0.0] * 3])}}


@pytest.fixture
def scenario(disciplines, design_space, uncertain_space, algo_data) -> UDOEScenario:
    """A DOE-based u-scenario."""
    scn = UDOEScenario(
        disciplines,
        "MDF",
        "f",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation="ControlVariate",
        statistic_estimation_parameters={
            "algo": "CustomDOE",
            "n_samples": None,
            "algo_options": {"samples": array([[0.0] * 3, [1.0] * 3])},
        },
    )
    scn.add_constraint("c", "Margin", factor=3.0)
    scn.add_observable("o", "Variance")
    scn.execute(algo_data)
    return scn


def test_scenario_optimum(scenario):
    """Check the optimum returned by the UDOEScenario."""
    assert_equal(scenario.optimization_result.x_opt, array([0.0] * 3))
    assert_allclose(scenario.optimization_result.f_opt, array([-12.0]), atol=1e-6)


def test_scenario_serialization(scenario, tmp_path, algo_data):
    """Check the serialization of a UDOEScenario."""
    file_path = tmp_path / "scenario.h5"
    scenario.to_pickle(file_path)
    saved_scn = UDOEScenario.from_pickle(file_path)
    saved_scn.execute(algo_data)
    assert_equal(scenario.optimization_result.x_opt, array([0.0] * 3))
    assert_allclose(scenario.optimization_result.f_opt, array([-12.0]), atol=1e-6)
    assert_equal(saved_scn.optimization_result.x_opt, array([0.0] * 3))
    assert_allclose(saved_scn.optimization_result.f_opt, array([-12.0]), atol=1e-6)


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


def test_umdo_formulation_objective(umdo_formulation):
    """Check that the UMDO formulation can compute the objective correctly."""
    objective = umdo_formulation.optimization_problem.objective
    assert_allclose(objective.evaluate(array([0.0] * 3)), array([-12.0]), atol=1e-6)


def test_umdo_formulation_constraint(umdo_formulation):
    """Check that the UMDO formulation can compute the constraints correctly."""
    constraint = umdo_formulation.optimization_problem.constraints[0]
    assert_allclose(constraint.evaluate(array([0.0] * 3)), array([-11.0]), atol=1e-6)


def test_umdo_formulation_observable(umdo_formulation):
    """Check that the UMDO formulation can compute the observables correctly."""
    observable = umdo_formulation.optimization_problem.observables[0]
    assert_allclose(observable.evaluate(array([0.0] * 3)), array([-10.0]))


def test_clear_inner_database(umdo_formulation):
    """Check that the inner database is cleared before sampling."""
    obj_value = umdo_formulation.optimization_problem.objective.evaluate(
        array([0.0] * 3)
    )
    # The inner problem depending on the uncertain variables is reset
    # when the outer problem changes the values of the design variables
    # to avoid recovering the data stored in the inner database
    # and force new evaluations of the functions attached to the inner problem.
    assert (
        umdo_formulation.optimization_problem.objective.evaluate(array([1.0, 0.0, 0.0]))
        != obj_value
    )


U_SAMPLES = array([[0.5, 1.0, 1.5], [-0.5, -1.0, -1.5]])
MEAN = array([1.0, 2.0])
JAC = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_estimate_mean(umdo_formulation):
    """Check the estimation of the mean."""
    statistic_function = Mean(umdo_formulation.uncertain_space)
    assert_almost_equal(
        statistic_function(array([[0.0, 0.0], [1.0, 2.0]]), U_SAMPLES, MEAN, JAC),
        array([-0.5, -1.0]),
    )


def test_estimate_variance(umdo_formulation):
    """Check the estimation of the variance."""
    statistic_function = Variance(umdo_formulation.uncertain_space)
    assert_almost_equal(
        statistic_function(array([[0.0, 0.0], [1.0, 2.0]]), U_SAMPLES, MEAN, JAC),
        array([0.263655, 1.267923]),
        decimal=6,
    )


def test_estimate_standard_derivation(umdo_formulation):
    """Check the estimation of the standard deviation."""
    statistic_function = StandardDeviation(umdo_formulation.uncertain_space)
    assert_almost_equal(
        statistic_function(array([[0.0, 0.0], [1.0, 2.0]]), U_SAMPLES, MEAN, JAC),
        array([0.263655, 1.267923]) ** 0.5,
        decimal=6,
    )


def test_estimate_margin(umdo_formulation):
    """Check the estimation of the margin."""
    statistic_function = Margin(umdo_formulation.uncertain_space, factor=3)
    assert_almost_equal(
        statistic_function(array([[0.0, 0.0], [1.0, 2.0]]), U_SAMPLES, MEAN, JAC),
        array([-0.5, -1.0]) + 3 * array([0.263655, 1.267923]) ** 0.5,
        decimal=6,
    )


@pytest.mark.parametrize(
    ("greater", "result"), [(False, array([1.0, 0.5])), (True, array([0.0, 0.5]))]
)
def test_estimate_probability(umdo_formulation, greater, result):
    """Check the estimation of the probability."""
    statistic_function = Probability(
        umdo_formulation.uncertain_space, threshold=1.5, greater=greater
    )
    assert_equal(
        statistic_function(array([[0.0, 0.0], [1.0, 2.0]]), U_SAMPLES, MEAN, JAC),
        result,
    )
