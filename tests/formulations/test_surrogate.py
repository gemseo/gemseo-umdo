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
from gemseo import execute_algo
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.doe.openturns.openturns import OpenTURNS
from gemseo.algos.doe.openturns.settings.ot_halton import OT_HALTON_Settings
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt
from gemseo.mlearning.regression.algos.linreg_settings import LinearRegressor_Settings
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from gemseo.mlearning.regression.algos.rbf_settings import RBFRegressor_Settings
from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_problem import IshigamiProblem
from gemseo.problems.uncertainty.utils import UniformDistribution
from gemseo.utils.seeder import SEED
from numpy import array
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.formulations.surrogate import Surrogate
from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario

if TYPE_CHECKING:
    from gemseo.algos.doe.base_doe_settings import BaseDOESettings
    from gemseo.core.mdo_functions.collections.observables import Observables
    from gemseo.typing import RealArray


@pytest.fixture(scope="module")
def ishigami_problem() -> IshigamiProblem:
    return IshigamiProblem(UniformDistribution.OPENTURNS)


@pytest.fixture(scope="module")
def rbf_regressor(ishigami_problem) -> RBFRegressor:
    """A RBF regressor for the Ishigami function."""
    execute_algo(ishigami_problem, algo_name="OT_HALTON", algo_type="doe", n_samples=20)
    learning_dataset = ishigami_problem.to_dataset(opt_naming=False)
    learning_dataset.rename_variable("Ishigami", "y")
    regressor = RBFRegressor(learning_dataset)
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
def umdo_formulation(ishigami_problem, doe_settings):
    """The UMDO formulation."""
    discipline = IshigamiDiscipline()
    formulation = Surrogate(
        [discipline],
        "y",
        DesignSpace(),
        DisciplinaryOpt([discipline], "y", ishigami_problem.design_space),
        ishigami_problem.design_space,
        "Mean",
        Surrogate_Settings(regressor_n_samples=10, doe_algo_settings=doe_settings),
    )
    formulation.add_constraint("y", "StandardDeviation")
    formulation.add_observable("y", "Variance")
    formulation.add_observable("y", "Margin", factor=3)
    formulation.add_observable("y", "Probability", greater=False, threshold=0.5)
    return formulation


@pytest.fixture(scope="module")
def output_samples(umdo_formulation, rbf_regressor) -> RealArray:
    """The output samples."""
    uncertain_space = umdo_formulation.uncertain_space
    convert_array_to_dict = uncertain_space.convert_array_to_dict
    doe_algo = DOELibraryFactory().create("MC")
    input_samples = doe_algo.compute_doe(uncertain_space, n_samples=10, seed=SEED)
    return rbf_regressor.predict(convert_array_to_dict(input_samples))["y"]


@pytest.fixture(scope="module")
def observables(umdo_formulation) -> Observables:
    """The observable functions."""
    return umdo_formulation.optimization_problem.observables


def test_output_samples(output_samples):
    """Check the shape of output_samples."""
    assert output_samples.shape == (10, 1)


_X = array([0.0])


def test_mean(umdo_formulation, output_samples):
    """Check the estimation of the mean from a surrogate-based UMDO formulation."""
    mean = output_samples.mean(0)
    assert_equal(umdo_formulation.optimization_problem.objective.evaluate(_X), mean)


def test_standard_deviation(umdo_formulation, output_samples):
    """Check the estimation of the std from a surrogate-based UMDO formulation."""
    std = output_samples.std(0, ddof=1)
    constraint = umdo_formulation.optimization_problem.constraints[0]
    assert_equal(constraint.evaluate(_X), std)


def test_variance(observables, output_samples):
    """Check the estimation of the variance from a surrogate-based UMDO formulation."""
    var = output_samples.var(0, ddof=1)
    assert_equal(observables[0].evaluate(_X), var)


def test_margin(observables, output_samples):
    """Check the estimation of a margin from a surrogate-based UMDO formulation."""
    mean = output_samples.mean(0)
    std = output_samples.std(0, ddof=1)
    assert_equal(observables[1].evaluate(_X), mean + 3 * std)


def test_probability(observables, output_samples):
    """Check the estimation of a probability from a surrogate-based UMDO formulation."""
    prob = (output_samples < 0.5).mean(0)
    assert_equal(observables[2].evaluate(_X), prob)


@pytest.mark.parametrize(
    ("statistic_estimation_parameters", "y_opt"),
    [
        ({"n_samples": 20}, 1.9689592736443002),
        ({"n_samples": 20, "regressor_n_samples": 10}, 2.4695858515160123),
        (
            {
                "n_samples": 20,
                "regressor_settings": RBFRegressor_Settings(function="cubic"),
            },
            2.017745497698664,
        ),
        (
            {"n_samples": 20, "regressor_settings": LinearRegressor_Settings()},
            1.9935244102531822,
        ),
    ],
)
def test_scenario(quadratic_problem, statistic_estimation_parameters, y_opt):
    """Check the surrogate-based U-MDO formulation in a scenario with a toy case."""
    discipline, design_space, uncertain_space = quadratic_problem
    scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="DisciplinaryOpt",
        statistic_estimation_settings=Surrogate_Settings(
            **statistic_estimation_parameters
        ),
    )
    scenario.execute(algo_name="CustomDOE", samples=array([[1.0]]))
    assert_almost_equal(scenario.optimization_result.x_opt, array([1.0]))
    assert_almost_equal(scenario.optimization_result.f_opt, y_opt)
    last_item = scenario.formulation.optimization_problem.database.last_item
    assert last_item.keys() == {"y_learning_quality", "y_test_quality", "E[y]"}
    assert last_item["y_learning_quality"].shape == (1,)
    assert last_item["y_test_quality"].shape == (1,)
