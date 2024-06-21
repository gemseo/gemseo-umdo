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
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt
from gemseo.mlearning.regression.algos.pce import PCERegressor
from gemseo.mlearning.regression.quality.r2_measure import R2Measure
from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_problem import IshigamiProblem
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from numpy import array
from numpy import full
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.formulations._statistics.pce.margin import Margin
from gemseo_umdo.formulations._statistics.pce.mean import Mean
from gemseo_umdo.formulations._statistics.pce.standard_deviation import (
    StandardDeviation,
)
from gemseo_umdo.formulations._statistics.pce.variance import Variance
from gemseo_umdo.formulations.pce import PCE
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

if TYPE_CHECKING:
    from gemseo.typing import NumberArray
    from gemseo.typing import RealArray


@pytest.fixture(scope="module")
def ishigami_problem() -> IshigamiProblem:
    return IshigamiProblem(IshigamiSpace.UniformDistribution.OPENTURNS)


@pytest.fixture(scope="module")
def pce_regressor(ishigami_problem) -> PCERegressor:
    """A PCE regressor for the Ishigami function."""
    execute_algo(ishigami_problem, "OT_HALTON", algo_type="doe", n_samples=20)
    learning_dataset = ishigami_problem.to_dataset(opt_naming=False)
    regressor = PCERegressor(learning_dataset, ishigami_problem.design_space)
    regressor.learn()
    return regressor


@pytest.fixture(scope="module")
def samples(ishigami_problem) -> RealArray:
    lib = OpenTURNS("OT_HALTON")
    return lib.compute_doe(ishigami_problem.design_space, 20)


@pytest.fixture(scope="module", params=("CustomDOE", "OT_HALTON"))
def doe_settings(request, samples) -> dict[str, str | int | dict[str, RealArray]]:
    if request.param == "CustomDOE":
        return {
            "doe_algo": "CustomDOE",
            "doe_algo_options": {"samples": samples},
        }

    return {"doe_algo": "OT_HALTON", "doe_n_samples": 20}


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
        **doe_settings,
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


def test_mean(pce_regressor, mean):
    """Check the PCE-based estimator of the mean."""
    assert_equal(Mean()(mean), mean)


def test_standard_deviation(pce_regressor, standard_deviation):
    """Check the PCE-based estimator of the standard deviation."""
    assert_equal(
        StandardDeviation()(standard_deviation),
        standard_deviation,
    )


def test_variance(pce_regressor, variance):
    """Check the PCE-based estimator of the variance."""
    assert_equal(Variance()(variance), variance)


def test_margin(pce_regressor, mean, standard_deviation):
    """Check the PCE-based estimator of the margin."""
    assert_equal(
        Margin()(mean, standard_deviation),
        mean + 2 * standard_deviation,
    )
    assert_equal(
        Margin(3)(mean, standard_deviation),
        mean + 3 * standard_deviation,
    )


def test_formulation(umdo_formulation, pce_regressor):
    """Check the estimation of statistics from a PCE-based UMDO formulation."""
    problem = umdo_formulation.optimization_problem
    x = array([0])
    mean = pce_regressor.mean
    assert_equal(problem.objective(x), mean)
    standard_deviation = pce_regressor.standard_deviation
    assert_equal(problem.constraints[0](x), standard_deviation)
    assert_equal(problem.observables[0](x), pce_regressor.variance)
    assert_equal(problem.observables[1](x), mean + 3 * standard_deviation)


def test_missing_n_samples(pce_regressor, ishigami_problem):
    """Check that an error is raised when the number of samples is missing."""
    discipline = IshigamiDiscipline()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The doe_n_samples argument of the U-MDO formulation 'PCE' is required."
        ),
    ):
        PCE(
            [discipline],
            "y",
            DesignSpace(),
            DisciplinaryOpt([discipline], "y", ishigami_problem.design_space),
            ishigami_problem.design_space,
            "Mean",
            doe_algo="OT_HALTON",
        )


def test_quality(caplog, pce_regressor, ishigami_problem):
    """Check that the PCE quality is logged."""
    discipline = IshigamiDiscipline()
    pce = PCE(
        [discipline],
        "y",
        DesignSpace(),
        DisciplinaryOpt([discipline], "y", ishigami_problem.design_space),
        ishigami_problem.design_space,
        "Mean",
        doe_algo="OT_HALTON",
        doe_n_samples=20,
    )
    pce.optimization_problem.objective(array([0]))
    module, level, message = caplog.record_tuples[0]
    assert module == "gemseo_umdo.formulations._functions.statistic_function_for_pce"
    assert level == logging.INFO
    regex = r" {8}R2Measure"
    assert re.compile(regex).match(message)

    module, level, message = caplog.record_tuples[1]
    assert module == "gemseo_umdo.formulations._functions.statistic_function_for_pce"
    assert level == logging.WARNING
    regex = r" {12}y\[0\]: \d+\.\d+<0\.9 \(train\) - -\d+\.\d+<0\.8 \(test\)"
    assert re.match(regex, message)


@pytest.mark.parametrize(
    ("quality_cv_compute", "regex"),
    [
        (False, r" {12}y\[0\]: \d+\.\d+>0\.9 \(train\)"),
        (True, r" {12}y\[0\]: \d+\.\d+>0\.9 \(train\) - \d+\.\d+>0\.8 \(test\)"),
    ],
)
def test_quality_cv(caplog, pce_regressor, ishigami_problem, quality_cv_compute, regex):
    """Check that the PCE quality with and without cross-validation and custom name."""
    discipline = IshigamiDiscipline()
    pce = PCE(
        [discipline],
        "y",
        DesignSpace(),
        DisciplinaryOpt([discipline], "y", ishigami_problem.design_space),
        ishigami_problem.design_space,
        "Mean",
        doe_algo="OT_HALTON",
        doe_n_samples=20,
        quality_name="MSEMeasure",
        quality_cv_compute=quality_cv_compute,
    )
    pce.optimization_problem.objective(array([0]))
    module, level, message = caplog.record_tuples[1]
    assert module == "gemseo_umdo.formulations._functions.statistic_function_for_pce"
    assert level == logging.WARNING
    assert re.match(regex, message)


def test_quality_cv_options(pce_regressor, ishigami_problem):
    """Check that the PCE quality options."""
    discipline = IshigamiDiscipline()
    with mock.patch.object(R2Measure, "compute_cross_validation_measure") as compute:
        pce = PCE(
            [discipline],
            "y",
            DesignSpace(),
            DisciplinaryOpt([discipline], "y", ishigami_problem.design_space),
            ishigami_problem.design_space,
            "Mean",
            doe_algo="OT_HALTON",
            doe_n_samples=20,
            quality_cv_n_folds=3,
            quality_cv_randomize=False,
            quality_cv_seed=12,
        )
        compute.return_value = {"y": array([0])}
        pce.optimization_problem.objective(array([0]))

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
            r" {12}y\[0\]: \d+\.\d+<0\.9 \(train\) - -\d+\.\d+<0\.9 \(test\)",
            True,
        ),
        (
            0.3,
            {"y": [0.9]},
            logging.WARNING,
            r" {12}y\[0\]: \d+\.\d+>=0\.3 \(train\) - -\d+\.\d+<0\.9 \(test\)",
            True,
        ),
        (
            {"y": 0.9},
            -2.0,
            logging.WARNING,
            r" {12}y\[0\]: \d+\.\d+<0\.9 \(train\) - -\d+\.\d+>=-2.0 \(test\)",
            True,
        ),
        (
            0.3,
            -2.0,
            logging.INFO,
            r" {12}y\[0\]: \d+\.\d+>=0\.3 \(train\) - -\d+\.\d+>=-2.0 \(test\)",
            True,
        ),
        (0.9, 0.9, logging.WARNING, r" {12}y\[0\]: \d+\.\d+<0\.9 \(train\)", False),
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
    pce = PCE(
        [discipline],
        "y",
        DesignSpace(),
        DisciplinaryOpt([discipline], "y", ishigami_problem.design_space),
        ishigami_problem.design_space,
        "Mean",
        doe_algo="OT_HALTON",
        doe_n_samples=20,
        quality_threshold=threshold,
        quality_cv_threshold=cv_threshold,
        quality_cv_compute=cv_compute,
    )
    pce.optimization_problem.objective(array([0]))
    _, level, message = caplog.record_tuples[1]
    assert level == expected_level
    assert re.match(regex, message)


def test_scenario():
    """Check the PCE-based U-MDO formulation in a scenario with a toy case."""
    discipline = AnalyticDiscipline({"y": "(x+u)**2"}, name="quadratic_function")

    design_space = DesignSpace()
    design_space.add_variable("x", l_b=-1, u_b=1.0, value=0.5)

    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("u", "OTNormalDistribution")

    scenario = UMDOScenario(
        [discipline],
        "DisciplinaryOpt",
        "y",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation="PCE",
        statistic_estimation_parameters={"doe_n_samples": 20},
    )
    scenario.execute({"algo": "NLOPT_COBYLA", "max_iter": 100})
    assert_almost_equal(scenario.optimization_result.x_opt, array([0.0]))
    assert_almost_equal(scenario.optimization_result.f_opt, array([1.0]))
