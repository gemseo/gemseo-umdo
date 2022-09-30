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
from gemseo_umdo.estimators.sampling import Margin
from gemseo_umdo.estimators.sampling import Mean
from gemseo_umdo.estimators.sampling import Probability
from gemseo_umdo.estimators.sampling import StandardDeviation
from gemseo_umdo.estimators.sampling import Variance
from gemseo_umdo.formulations.sampling import Sampling
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario
from numpy import array
from numpy import ndarray
from numpy.testing import assert_equal


@pytest.fixture
def umdo_formulation(
    disciplines: Sequence[MDODiscipline],
    design_space: DesignSpace,
    mdo_formulation: MDF,
    uncertain_space: ParameterSpace,
) -> Sampling:
    """The UMDO formulation."""
    design_space = MDF(disciplines, "f", design_space).design_space
    formulation = Sampling(
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


@pytest.fixture
def mdo_samples(mdf_discipline) -> list[dict[str, ndarray]]:
    """The samples of the MDO formulations at x = [0,0,0] and x = [1,1,1]."""
    return [
        mdf_discipline.execute({name: array([i]) for name in ["u", "u1", "u2"]})
        for i in [0.0, 1.0]
    ]


def test_scenario(disciplines, design_space, uncertain_space, tmp_path):
    """Check the optimum returned by the UMDOScenario."""
    scn = UDOEScenario(
        disciplines,
        "MDF",
        "f",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation="Sampling",
        statistic_estimation_parameters={
            "algo": "CustomDOE",
            "n_samples": None,
            "algo_options": {"samples": array([[0.0] * 3, [1.0] * 3])},
        },
    )
    scn.add_constraint("c", "Margin", factor=3.0)
    scn.add_observable("o", "Variance")
    file_path = tmp_path / "scenario.h5"
    scn.serialize(file_path)
    algo_data = {"algo": "CustomDOE", "algo_options": {"samples": array([[0.0] * 3])}}
    scn.execute(algo_data)
    saved_scn = UDOEScenario.deserialize(file_path)
    saved_scn.execute(algo_data)
    assert_equal(scn.optimization_result.x_opt, array([0.0] * 3))
    assert scn.optimization_result.f_opt == -2.0
    assert_equal(saved_scn.optimization_result.x_opt, array([0.0] * 3))
    assert saved_scn.optimization_result.f_opt == -2.0


def test_estimate_mean(umdo_formulation):
    """Check the estimation of the mean."""
    mean_estimation = Mean(umdo_formulation)(array([[0.0, 0.0], [1.0, 2.0]]))
    assert_equal(mean_estimation, array([0.5, 1.0]))


def test_estimate_variance(umdo_formulation):
    """Check the estimation of the variance."""
    var_estimation = Variance(umdo_formulation)(array([[0.0, 0.0], [1.0, 2.0]]))
    assert_equal(var_estimation, array([0.25, 1.0]))


def test_estimate_standard_derivation(umdo_formulation):
    """Check the estimation of the standard deviation."""
    std_estimation = StandardDeviation(umdo_formulation)(
        array([[0.0, 0.0], [1.0, 2.0]])
    )
    expected = Variance(umdo_formulation)(array([[0.0, 0.0], [1.0, 2.0]])) ** 0.5
    assert_equal(std_estimation, expected)


def test_estimate_margin(umdo_formulation):
    """Check the estimation of the margin."""
    data = array([[0.0, 0.0], [1.0, 2.0]])
    expected = Mean(umdo_formulation)(data) + 3.0 * StandardDeviation(umdo_formulation)(
        data
    )
    margin_estimation = Margin(umdo_formulation)(data, factor=3.0)
    assert_equal(margin_estimation, expected)


@pytest.mark.parametrize(
    "greater,result", [(False, array([1.0, 1.0])), (True, array([0.0, 0.5]))]
)
def test_estimate_probability(umdo_formulation, greater, result):
    """Check the estimation of the probability."""
    probability_estimation = Probability(umdo_formulation)(
        array([[0.0, 0.0], [1.0, 2.0]]), 2.0, greater=greater
    )
    assert_equal(probability_estimation, result)


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


def test_umdo_formulation_objective(umdo_formulation, mdo_samples):
    """Check that the UMDO formulation can compute the objective correctly."""
    objective = umdo_formulation.opt_problem.objective
    assert_equal(
        objective(array([0.0] * 3)),
        sum(mdo_sample["f"][0] for mdo_sample in mdo_samples) / 2,
    )


def test_umdo_formulation_constraint(umdo_formulation, mdo_samples):
    """Check that the UMDO formulation can compute the constraints correctly."""
    constraint = umdo_formulation.opt_problem.constraints[0]
    assert_equal(
        constraint(array([0.0] * 3)),
        sum(mdo_sample["c"][0] for mdo_sample in mdo_samples) / 2,
    )


def test_umdo_formulation_observable(umdo_formulation, mdo_samples):
    """Check that the UMDO formulation can compute the observables correctly."""
    observable = umdo_formulation.opt_problem.observables[0]
    assert_equal(
        observable(array([0.0] * 3)),
        sum(mdo_sample["o"][0] for mdo_sample in mdo_samples) / 2,
    )


def test_clear_inner_database(umdo_formulation):
    """Check that the inner database is cleared before sampling."""
    assert "f" not in umdo_formulation._processed_functions
    obj_value = umdo_formulation.opt_problem.objective(array([0.0] * 3))
    assert "f" in umdo_formulation._processed_functions
    # The inner problem depending on the uncertain variables is reset
    # when the outer problem changes the values of the design variables
    # to avoid recovering the data stored in the inner database
    # and force new evaluations of the functions attached to the inner problem.
    assert umdo_formulation.opt_problem.objective(array([1.0, 0.0, 0.0])) != obj_value


def test_read_write_n_samples(umdo_formulation):
    """Check the property and setter _n_samples."""
    doe_algo_options = umdo_formulation._Sampling__doe_algo_options

    # Sampling has been instantiated with `n_samples=None`.
    assert umdo_formulation._n_samples is None
    assert doe_algo_options["n_samples"] is None

    # In the options of the DOE,
    # the number of samples is set to 3 with the property _n_samples.
    umdo_formulation._n_samples = 3
    assert umdo_formulation._n_samples == doe_algo_options["n_samples"] == 3
