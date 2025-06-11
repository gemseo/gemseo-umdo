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

import re
from typing import TYPE_CHECKING
from typing import Any

import pytest
from gemseo import from_pickle
from gemseo import to_pickle
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.mdf import MDF
from gemseo.mlearning.regression.algos.rbf_settings import RBFRegressor_Settings
from numpy import array
from numpy import diag
from numpy import diagonal
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
from gemseo_umdo.formulations.control_variate_settings import ControlVariate_Settings
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline.discipline import Discipline
    from gemseo.mlearning.regression.algos.base_regressor_settings import (
        BaseRegressorSettings,
    )


@pytest.fixture(params=[None, RBFRegressor_Settings()])
def regressor_settings(request) -> BaseRegressorSettings | None:
    """The regressor settings if any."""
    return request.param


@pytest.fixture
def umdo_formulation(
    disciplines: Sequence[Discipline],
    design_space: DesignSpace,
    mdo_formulation: MDF,
    uncertain_space: ParameterSpace,
    regressor_settings: BaseRegressorSettings,
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
        settings_model=ControlVariate_Settings(
            doe_algo_settings=CustomDOE_Settings(samples=array([[0.0] * 3, [1.0] * 3])),
            regressor_settings=regressor_settings,
        ),
    )
    formulation.add_constraint("c", "Mean")
    formulation.add_observable("o", "Mean")
    return formulation


@pytest.fixture(scope="module")
def algo_data() -> dict[str, Any]:
    """Input data for a DOE-based u-scenario."""
    return {"algo_name": "CustomDOE", "samples": array([[0.0] * 3])}


@pytest.fixture
def scenario(disciplines, design_space, uncertain_space, algo_data) -> UDOEScenario:
    """A DOE-based u-scenario."""
    scn = UDOEScenario(
        disciplines,
        "f",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="MDF",
        statistic_estimation_settings=ControlVariate_Settings(
            doe_algo_settings=CustomDOE_Settings(samples=array([[0.0] * 3, [1.0] * 3])),
        ),
    )
    scn.add_constraint("c", "Margin", factor=3.0)
    scn.add_observable("o", "Variance")
    scn.execute(**algo_data)
    return scn


def test_scenario_optimum(scenario):
    """Check the optimum returned by the UDOEScenario."""
    assert_equal(scenario.optimization_result.x_opt, array([0.0] * 3))
    assert_allclose(scenario.optimization_result.f_opt, array([-12.0]), atol=1e-6)


def test_scenario_serialization(scenario, tmp_path, algo_data):
    """Check the serialization of a UDOEScenario."""
    file_path = tmp_path / "scenario.h5"
    to_pickle(scenario, file_path)
    saved_scn = from_pickle(file_path)
    saved_scn.execute(**algo_data)
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
    expected = -12.0 if umdo_formulation._settings.regressor_settings is None else -2.0
    assert_allclose(objective.evaluate(array([0.0] * 3)), array([expected]), atol=1e-6)


def test_umdo_formulation_constraint(umdo_formulation):
    """Check that the UMDO formulation can compute the constraints correctly."""
    constraint = umdo_formulation.optimization_problem.constraints[0]
    expected = -11.0 if umdo_formulation._settings.regressor_settings is None else -1.5
    assert_allclose(constraint.evaluate(array([0.0] * 3)), array([expected]), atol=1e-6)


def test_umdo_formulation_observable(umdo_formulation):
    """Check that the UMDO formulation can compute the observables correctly."""
    observable = umdo_formulation.optimization_problem.observables[0]
    expected = -10.0 if umdo_formulation._settings.regressor_settings is None else -1.0
    assert_allclose(observable.evaluate(array([0.0] * 3)), array([expected]))


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
VARIANCE = diagonal(JAC @ diag(array([1.0, 2.0, 1.5])) @ JAC.T)


def test_estimate_mean(umdo_formulation):
    """Check the estimation of the mean."""
    statistic_function = Mean(umdo_formulation.uncertain_space)
    assert_almost_equal(
        statistic_function.estimate_statistic(
            array([[0.0, 0.0], [1.0, 2.0]]),
            MEAN,
            VARIANCE,
            array([[1.0, 1.1], [2.0, 2.1]]),
            array([[1.3, 1.4], [2.6, 2.7]]),
        ),
        array([0.0, 1.8]),
    )


def test_estimate_variance(umdo_formulation):
    """Check the estimation of the variance."""
    statistic_function = Variance(umdo_formulation.uncertain_space)
    assert_almost_equal(
        statistic_function.estimate_statistic(
            array([[0.0, 0.0], [1.0, 2.0]]),
            MEAN,
            VARIANCE,
            array([[1.0, 1.1], [2.0, 2.1]]),
            array([[1.3, 1.4], [2.6, 2.7]]),
        ),
        array([0.25, 1.0]),
        decimal=6,
    )


def test_estimate_standard_derivation(umdo_formulation):
    """Check the estimation of the standard deviation."""
    statistic_function = StandardDeviation(umdo_formulation.uncertain_space)
    assert_almost_equal(
        statistic_function.estimate_statistic(
            array([[0.0, 0.0], [1.0, 2.0]]),
            MEAN,
            VARIANCE,
            array([[1.0, 1.1], [2.0, 2.1]]),
            array([[1.3, 1.4], [2.6, 2.7]]),
        ),
        array([0.25, 1.0]) ** 0.5,
        decimal=6,
    )


def test_estimate_margin(umdo_formulation):
    """Check the estimation of the margin."""
    statistic_function = Margin(umdo_formulation.uncertain_space, factor=3)
    assert_almost_equal(
        statistic_function.estimate_statistic(
            array([[0.0, 0.0], [1.0, 2.0]]),
            MEAN,
            VARIANCE,
            array([[1.0, 1.1], [2.0, 2.1]]),
            array([[1.3, 1.4], [2.6, 2.7]]),
        ),
        array([0.0, 1.8]) + 3 * array([0.25, 1.0]) ** 0.5,
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
        statistic_function.estimate_statistic(
            array([[0.0, 0.0], [1.0, 2.0]]),
            MEAN,
            VARIANCE,
            array([[1.0], [2.0]]),
            array([[1.3], [2.6]]),
        ),
        result,
    )


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
        statistic_estimation_settings=ControlVariate_Settings(n_samples=2),
        formulation_name="DisciplinaryOpt",
    )
    scenario.execute(CustomDOE_Settings(samples=array([[1.0]])))
    assert_almost_equal(discipline.io.data["x"], array([1.0]))
    # u = 1.125, f = 2.125 and dfdu = 1. before bug fix
    assert_almost_equal(discipline.io.data["u"], array([0.75]))
    assert_almost_equal(discipline.io.data["f"], array([1.75]))


def test_seeds_validator():
    """Verify that the seeds validator of ControlVariate_Settings works correctly."""
    ControlVariate_Settings(
        doe_algo_settings=MC_Settings(seed=1, n_samples=10),
        regressor_doe_algo_settings=MC_Settings(seed=2, n_samples=10),
    )
    ControlVariate_Settings(
        doe_algo_settings=MC_Settings(seed=1, n_samples=10),
        regressor_doe_algo_settings=OT_OPT_LHS_Settings(seed=1, n_samples=10),
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The seed for sampling "
            "and the seed for creating the training dataset must be different; "
            "both are 1."
        ),
    ):
        ControlVariate_Settings(
            doe_algo_settings=MC_Settings(seed=1, n_samples=10),
            regressor_doe_algo_settings=MC_Settings(seed=1, n_samples=10),
        )
