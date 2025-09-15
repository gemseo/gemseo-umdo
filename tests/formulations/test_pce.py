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

import logging
import re
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from gemseo import execute_algo
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.doe.openturns.openturns import OpenTURNS
from gemseo.algos.doe.openturns.settings.ot_halton import OT_HALTON_Settings
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt
from gemseo.mlearning.regression.algos.fce_settings import FCERegressor_Settings
from gemseo.mlearning.regression.algos.pce import PCERegressor
from gemseo.mlearning.regression.algos.pce_settings import PCERegressor_Settings
from gemseo.mlearning.regression.quality.r2_measure import R2Measure
from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_problem import IshigamiProblem
from gemseo.problems.uncertainty.utils import UniformDistribution
from numpy import array
from numpy import full
from numpy import ones
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.formulations._statistics.pce.margin import Margin
from gemseo_umdo.formulations._statistics.pce.mean import Mean
from gemseo_umdo.formulations._statistics.pce.standard_deviation import (
    StandardDeviation,
)
from gemseo_umdo.formulations._statistics.pce.variance import Variance
from gemseo_umdo.formulations.pce import PCE
from gemseo_umdo.formulations.pce_settings import PCE_Settings
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario

if TYPE_CHECKING:
    from gemseo.algos.doe.base_doe_settings import BaseDOESettings
    from gemseo.core.mdo_functions.collections.observables import Observables
    from gemseo.typing import NumberArray
    from gemseo.typing import RealArray


@pytest.fixture(scope="module")
def ishigami_problem() -> IshigamiProblem:
    return IshigamiProblem(UniformDistribution.OPENTURNS)


@pytest.fixture(scope="module")
def pce_regressor(ishigami_problem) -> PCERegressor:
    """A PCE regressor for the Ishigami function."""
    execute_algo(ishigami_problem, algo_name="OT_HALTON", algo_type="doe", n_samples=20)
    regressor = PCERegressor(ishigami_problem.to_dataset(opt_naming=False))
    regressor.learn()
    return regressor


@pytest.fixture(scope="module")
def samples(ishigami_problem) -> RealArray:
    lib = OpenTURNS("OT_HALTON")
    return lib.compute_doe(ishigami_problem.design_space, n_samples=20)


@pytest.fixture(scope="module", params=("CustomDOE", "OT_HALTON"))
def doe_settings(request, samples) -> BaseDOESettings:
    if request.param == "CustomDOE":
        return CustomDOE_Settings(samples=samples)

    return OT_HALTON_Settings(n_samples=20)


@pytest.fixture(scope="module")
def umdo_formulation(pce_regressor, ishigami_problem, doe_settings):
    """The UMDO formulation."""
    discipline = IshigamiDiscipline()
    formulation = PCE(
        [discipline],
        "y",
        DesignSpace(),
        DisciplinaryOpt([discipline], "y", ishigami_problem.design_space),
        ishigami_problem.design_space,
        "Mean",
        PCE_Settings(doe_algo_settings=doe_settings),
    )
    formulation.add_constraint("y", "StandardDeviation")
    formulation.add_observable("y", "Variance")
    formulation.add_observable("y", "Margin", factor=3)
    return formulation


@pytest.fixture(scope="module")
def mean() -> NumberArray:
    """A vector mocking the mean computed from the PCE coefficients."""
    return full(2, 0.1)


@pytest.fixture(scope="module")
def standard_deviation() -> NumberArray:
    """A vector mocking the standard deviation from the PCE coefficients."""
    return full(2, 0.2)


@pytest.fixture(scope="module")
def variance() -> NumberArray:
    """A vector mocking the variance from the PCE coefficients."""
    return full(2, 0.3)


def test_mean(mean):
    """Check the PCE-based estimator of the mean."""
    assert_equal(Mean().estimate_statistic(mean), mean)


def test_standard_deviation(standard_deviation):
    """Check the PCE-based estimator of the standard deviation."""
    assert_equal(
        StandardDeviation().estimate_statistic(standard_deviation),
        standard_deviation,
    )


def test_variance(variance):
    """Check the PCE-based estimator of the variance."""
    assert_equal(Variance().estimate_statistic(variance), variance)


def test_margin(mean, standard_deviation):
    """Check the PCE-based estimator of the margin."""
    assert_equal(
        Margin().estimate_statistic(mean, standard_deviation),
        mean + 2 * standard_deviation,
    )
    assert_equal(
        Margin(3).estimate_statistic(mean, standard_deviation),
        mean + 3 * standard_deviation,
    )


@pytest.fixture(scope="module")
def observables(umdo_formulation) -> Observables:
    """The observable functions."""
    return umdo_formulation.optimization_problem.observables


_X = array([0.0])


def test_mean_from_formulation(umdo_formulation, pce_regressor):
    """Check the estimation of the mean from a PCE-based UMDO formulation."""
    mean = pce_regressor.mean
    assert_equal(umdo_formulation.optimization_problem.objective.evaluate(_X), mean)


def test_std_from_formulation(umdo_formulation, pce_regressor):
    """Check the estimation of the std from a PCE-based UMDO formulation."""
    std = pce_regressor.standard_deviation
    assert_equal(umdo_formulation.optimization_problem.constraints[0].evaluate(_X), std)


def test_variance_from_formulation(observables, pce_regressor):
    """Check the estimation of the variance from a PCE-based UMDO formulation."""
    var = pce_regressor.variance
    assert_equal(observables[0].evaluate(_X), var)


def test_margin_from_formulation(observables, pce_regressor):
    """Check the estimation of a margin from a PCE-based UMDO formulation."""
    mean = pce_regressor.mean
    std = pce_regressor.standard_deviation
    assert_equal(observables[1].evaluate(_X), mean + 3 * std)


def test_quality(caplog, pce_regressor, ishigami_problem):
    """Check that the PCE quality is logged."""
    discipline = IshigamiDiscipline()
    design_space = ishigami_problem.design_space
    pce = PCE(
        [discipline],
        "y",
        DesignSpace(),
        DisciplinaryOpt([discipline], "y", design_space),
        design_space,
        "Mean",
        settings_model=PCE_Settings(doe_algo_settings=OT_HALTON_Settings(n_samples=20)),
    )
    pce.optimization_problem.objective.evaluate(array([0.0]))
    module, level, message = caplog.record_tuples[0]
    assert (
        module == "gemseo_umdo.formulations._functions.statistic_function_for_surrogate"
    )
    assert level == logging.INFO
    regex = r" {8}R2Measure"
    assert re.compile(regex).match(message)

    module, level, message = caplog.record_tuples[1]
    assert (
        module == "gemseo_umdo.formulations._functions.statistic_function_for_surrogate"
    )
    assert level == logging.WARNING
    regex = r" {12}y\[0\]: \d+\.\d+<0\.9 \(learning\) - -\d+\.\d+<0\.8 \(test\)"
    assert re.match(regex, message)


@pytest.mark.parametrize(
    ("quality_cv_compute", "regex"),
    [
        (False, r" {12}y\[0\]: \d+\.\d+>0\.9 \(learning\)"),
        (True, r" {12}y\[0\]: \d+\.\d+>0\.9 \(learning\) - \d+\.\d+>0\.8 \(test\)"),
    ],
)
def test_quality_cv(caplog, pce_regressor, ishigami_problem, quality_cv_compute, regex):
    """Check that the PCE quality with and without cross-validation and custom name."""
    discipline = IshigamiDiscipline()
    design_space = ishigami_problem.design_space
    pce = PCE(
        [discipline],
        "y",
        DesignSpace(),
        DisciplinaryOpt([discipline], "y", design_space),
        design_space,
        "Mean",
        settings_model=PCE_Settings(
            doe_algo_settings=OT_HALTON_Settings(n_samples=20),
            quality_name="MSEMeasure",
            quality_cv_compute=quality_cv_compute,
        ),
    )
    pce.optimization_problem.objective.evaluate(array([0.0]))
    module, level, message = caplog.record_tuples[1]
    assert (
        module == "gemseo_umdo.formulations._functions.statistic_function_for_surrogate"
    )
    assert level == logging.WARNING
    assert re.match(regex, message)


def test_quality_cv_options(pce_regressor, ishigami_problem):
    """Check that the PCE quality options."""
    discipline = IshigamiDiscipline()
    design_space = ishigami_problem.design_space
    with mock.patch.object(R2Measure, "compute_cross_validation_measure") as compute:
        pce = PCE(
            [discipline],
            "y",
            DesignSpace(),
            DisciplinaryOpt([discipline], "y", design_space),
            design_space,
            "Mean",
            settings_model=PCE_Settings(
                doe_algo_settings=OT_HALTON_Settings(n_samples=20),
                quality_cv_n_folds=3,
                quality_cv_randomize=False,
                quality_cv_seed=12,
            ),
        )
        compute.return_value = {"y": array([0.0])}
        pce.optimization_problem.objective.evaluate(array([0.0]))

    assert compute.call_args.kwargs == {
        "as_dict": True,
        "n_folds": 3,
        "randomize": False,
        "seed": 12,
    }


@pytest.mark.parametrize(
    ("threshold", "cv_threshold", "expected_level", "regex", "cv_compute"),
    [
        (
            0.9,
            0.9,
            logging.WARNING,
            r" {12}y\[0\]: \d+\.\d+<0\.9 \(learning\) - -\d+\.\d+<0\.9 \(test\)",
            True,
        ),
        (
            0.3,
            {"y": [0.9]},
            logging.WARNING,
            r" {12}y\[0\]: \d+\.\d+>=0\.3 \(learning\) - -\d+\.\d+<0\.9 \(test\)",
            True,
        ),
        (
            {"y": 0.9},
            -2.0,
            logging.WARNING,
            r" {12}y\[0\]: \d+\.\d+<0\.9 \(learning\) - -\d+\.\d+>=-2.0 \(test\)",
            True,
        ),
        (
            0.3,
            -2.0,
            logging.INFO,
            r" {12}y\[0\]: \d+\.\d+>=0\.3 \(learning\) - -\d+\.\d+>=-2.0 \(test\)",
            True,
        ),
        (0.9, 0.9, logging.WARNING, r" {12}y\[0\]: \d+\.\d+<0\.9 \(learning\)", False),
    ],
)
def test_quality_log_level(
    caplog,
    pce_regressor,
    ishigami_problem,
    threshold,
    cv_threshold,
    expected_level,
    regex,
    cv_compute,
):
    """Check that the log level of the PCE quality."""
    discipline = IshigamiDiscipline()
    design_space = ishigami_problem.design_space
    pce = PCE(
        [discipline],
        "y",
        DesignSpace(),
        DisciplinaryOpt([discipline], "y", design_space),
        design_space,
        "Mean",
        settings_model=PCE_Settings(
            doe_algo_settings=OT_HALTON_Settings(n_samples=20),
            quality_threshold=threshold,
            quality_cv_threshold=cv_threshold,
            quality_cv_compute=cv_compute,
        ),
    )
    pce.optimization_problem.objective.evaluate(array([0.0]))
    _, level, message = caplog.record_tuples[1]
    assert level == expected_level
    assert re.match(regex, message)


@pytest.mark.parametrize(
    ("input_dimension", "pce_settings", "regressor_settings", "f_opt", "jac_opt"),
    [
        (1, {}, PCERegressor_Settings(), 2.0, array([[2.0]])),
        (1, {}, PCERegressor_Settings(degree=1), 1.9972399330987038, array([[2.0]])),
        (
            1,
            {"approximate_statistics_jacobians": True},
            PCERegressor_Settings(),
            2.0,
            array([[2.0]]),
        ),
        (
            1,
            {"approximate_statistics_jacobians": True, "differentiation_step": 1e-2},
            PCERegressor_Settings(),
            2.0,
            array([[2.0]]),
        ),
        (2, {}, PCERegressor_Settings(), 5.916420174564085, array([[[2.0, 6.0]]])),
        (
            2,
            {},
            PCERegressor_Settings(degree=1),
            6.404476351650847,
            array([[[2.0, 6.5868963]]]),
        ),
        (
            2,
            {"approximate_statistics_jacobians": True},
            PCERegressor_Settings(),
            5.916420174564085,
            array([[[1.8819588, 7.0932709]]]),
        ),
        (
            2,
            {"approximate_statistics_jacobians": True, "differentiation_step": 1e-2},
            PCERegressor_Settings(),
            5.916420174564085,
            array([[[1.8819588, 7.0932709]]]),
        ),
        (
            1,
            {},
            FCERegressor_Settings(),
            2.0,
            array([[2.0]]),
        ),
    ],
)
def test_scenario(
    quadratic_problem,
    input_dimension,
    pce_settings,
    regressor_settings,
    f_opt,
    jac_opt,
):
    """Check the PCE-based U-MDO formulation in a scenario with a toy case."""
    discipline, design_space, uncertain_space = quadratic_problem
    if input_dimension == 2:
        design_space.add_variable("z", lower_bound=-1, upper_bound=1.0, value=0.5)
        uncertain_space.add_random_variable("v", "OTNormalDistribution")
    scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="DisciplinaryOpt",
        statistic_estimation_settings=PCE_Settings(
            n_samples=20,
            regressor_settings=regressor_settings,
            **pce_settings,
        ),
    )
    scenario.execute(
        algo_name="CustomDOE", samples=ones((1, input_dimension)), eval_jac=True
    )
    assert_almost_equal(scenario.optimization_result.x_opt, ones(input_dimension))
    assert_almost_equal(scenario.optimization_result.f_opt, f_opt)
    get = scenario.formulation.optimization_problem.database.get_gradient_history
    jac = get("E[y]")
    assert_almost_equal(jac, jac_opt)
