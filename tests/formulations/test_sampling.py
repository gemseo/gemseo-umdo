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

import pickle
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pytest
from gemseo.datasets.io_dataset import IODataset
from gemseo.formulations.mdf import MDF
from gemseo.utils.comparisons import compare_dict_of_arrays
from numpy import array
from numpy import ndarray
from numpy.testing import assert_equal
from pandas._testing import assert_frame_equal

from gemseo_umdo.formulations._statistics.iterative_sampling.margin import (
    Margin as IterativeMargin,
)
from gemseo_umdo.formulations._statistics.iterative_sampling.mean import (
    Mean as IterativeMean,
)
from gemseo_umdo.formulations._statistics.iterative_sampling.probability import (
    Probability as IterativeProbability,
)
from gemseo_umdo.formulations._statistics.iterative_sampling.standard_deviation import (
    StandardDeviation as IterativeStandardDeviation,
)
from gemseo_umdo.formulations._statistics.iterative_sampling.variance import (
    Variance as IterativeVariance,
)
from gemseo_umdo.formulations._statistics.sampling.base_sampling_estimator import (
    BaseSamplingEstimator,
)
from gemseo_umdo.formulations._statistics.sampling.margin import Margin
from gemseo_umdo.formulations._statistics.sampling.mean import Mean
from gemseo_umdo.formulations._statistics.sampling.probability import Probability
from gemseo_umdo.formulations._statistics.sampling.standard_deviation import (
    StandardDeviation,
)
from gemseo_umdo.formulations._statistics.sampling.variance import Variance
from gemseo_umdo.formulations.sampling import Sampling
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline

    from gemseo_umdo.formulations._statistics.iterative_sampling.sampling_estimator import (  # noqa: E501
        SamplingEstimator as IterativeSamplingEstimator,
    )


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


@pytest.fixture
def scenario_input_data() -> dict[str, str | dict[str, ndarray]]:
    """The input data of the scenario."""
    return {"algo": "CustomDOE", "algo_options": {"samples": array([[0.0] * 3])}}


@pytest.fixture(params=[False, True])
def estimate_statistics_iteratively(request) -> bool:
    """Whether to estimate the statistics iteratively."""
    return request.param


@pytest.fixture(params=[1, 2])
def n_processes(request) -> int:
    """Whether to estimate the statistics iteratively."""
    return request.param


@pytest.fixture(params=[False, True])
def maximize_objective(request) -> int:
    """Whether to maximize the objective."""
    return request.param


@pytest.fixture
def scenario(
    estimate_statistics_iteratively,
    n_processes,
    maximize_objective,
    disciplines,
    design_space,
    uncertain_space,
):
    """A scenario of interest."""
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
            "algo_options": {
                "samples": array([[0.0] * 3, [1.0] * 3]),
                "n_processes": n_processes,
            },
            "estimate_statistics_iteratively": estimate_statistics_iteratively,
        },
        maximize_objective=maximize_objective,
    )
    scn.add_constraint("c", "Margin", factor=3.0)
    scn.add_observable("o", "Variance")
    return scn


def test_scenario_execution(scenario, maximize_objective, scenario_input_data, caplog):
    """Check the execution of an UMDOScenario with the Sampling U-MDO formulation."""
    scenario.execute(scenario_input_data)
    assert_equal(scenario.optimization_result.x_opt, array([0.0, 0.0, 0.0]))
    expected_f_opt = 2.0 if maximize_objective else -2.0
    assert scenario.optimization_result.f_opt == expected_f_opt
    # Check that the sampling is not logged.
    assert "minimize f" not in caplog.text


def test_scenario_serialization(scenario, tmp_path, scenario_input_data):
    """Check the serialization of an UMDOScenario with Sampling U-MDO formulation."""
    file_path = tmp_path / "scenario.h5"
    scenario.to_pickle(file_path)
    saved_scenario = UDOEScenario.from_pickle(file_path)
    saved_scenario.execute(scenario_input_data)
    assert_equal(saved_scenario.optimization_result.x_opt, array([0.0, 0.0, 0.0]))


def estimate(
    statistic_class: BaseSamplingEstimator | IterativeSamplingEstimator,
    samples: ndarray,
    **options: Any,
) -> ndarray:
    """A function estimating a statistic.

    Args:
        statistic_class: The class of the statistic.
        samples: The samples to estimate the statistics.
        **options: The options to instantiate the class of the statistic.

    Returns:
        The estimation of the statistic.
    """
    statistic = statistic_class(**options)
    if issubclass(statistic_class, BaseSamplingEstimator):
        return statistic(samples)

    for sample in samples:
        result = statistic(sample)

    return result


@pytest.mark.parametrize("statistic_class", [Mean, IterativeMean])
def test_estimate_mean(statistic_class):
    """Check the estimation of the mean."""
    assert_equal(
        estimate(statistic_class, array([[0.0, 0.0], [1.0, 2.0]])), array([0.5, 1.0])
    )


@pytest.mark.parametrize("statistic_class", [Variance, IterativeVariance])
def test_estimate_variance(statistic_class):
    """Check the estimation of the variance."""
    assert_equal(
        estimate(statistic_class, array([[0.0, 0.0], [1.0, 2.0]])), array([0.5, 2.0])
    )


@pytest.mark.parametrize(
    "statistic_class", [StandardDeviation, IterativeStandardDeviation]
)
def test_estimate_standard_derivation(statistic_class):
    """Check the estimation of the standard deviation."""
    assert_equal(
        estimate(statistic_class, array([[0.0, 0.0], [1.0, 2.0]])),
        array([0.5, 2.0]) ** 0.5,
    )


@pytest.mark.parametrize("statistic_class", [Margin, IterativeMargin])
def test_estimate_margin(statistic_class):
    """Check the estimation of the margin."""
    assert_equal(
        estimate(statistic_class, array([[0.0, 0.0], [1.0, 2.0]]), factor=3),
        array([0.5, 1.0]) + 3 * array([0.5, 2.0]) ** 0.5,
    )


@pytest.mark.parametrize(
    ("greater", "result"), [(False, array([1.0, 0.5])), (True, array([0.0, 0.5]))]
)
@pytest.mark.parametrize("statistic_class", [Probability, IterativeProbability])
def test_estimate_probability(greater, result, statistic_class):
    """Check the estimation of the probability."""
    assert_equal(
        estimate(
            statistic_class,
            array([[0.0, 0.0], [1.0, 2.0]]),
            threshold=1.5,
            greater=greater,
        ),
        result,
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


def test_umdo_formulation_objective(umdo_formulation, mdo_samples):
    """Check that the UMDO formulation can compute the objective correctly."""
    objective = umdo_formulation.optimization_problem.objective
    assert_equal(
        objective.evaluate(array([0.0] * 3)),
        sum(mdo_sample["f"][0] for mdo_sample in mdo_samples) / 2,
    )


def test_umdo_formulation_constraint(umdo_formulation, mdo_samples):
    """Check that the UMDO formulation can compute the constraints correctly."""
    constraint = umdo_formulation.optimization_problem.constraints[0]
    assert_equal(
        constraint.evaluate(array([0.0] * 3)),
        sum(mdo_sample["c"][0] for mdo_sample in mdo_samples) / 2,
    )


def test_umdo_formulation_observable(umdo_formulation, mdo_samples):
    """Check that the UMDO formulation can compute the observables correctly."""
    observable = umdo_formulation.optimization_problem.observables[0]
    assert_equal(
        observable.evaluate(array([0.0] * 3)),
        sum(mdo_sample["o"][0] for mdo_sample in mdo_samples) / 2,
    )


def test_clear_inner_database(umdo_formulation):
    """Check that the inner database is cleared before sampling."""
    # The inner problem depending on the uncertain variables is reset
    # when the outer problem changes the values of the design variables
    # to avoid recovering the data stored in the inner database
    # and force new evaluations of the functions attached to the inner problem.
    obj_value = umdo_formulation.optimization_problem.objective.evaluate(
        array([0.0] * 3)
    )
    assert (
        umdo_formulation.optimization_problem.objective.evaluate(array([1.0, 0.0, 0.0]))
        != obj_value
    )


def test_read_write_n_samples(umdo_formulation):
    """Check the property and setter _n_samples."""
    doe_algo_options = umdo_formulation._Sampling__doe_algo_options

    # Sampling has been instantiated with `n_samples=None`.
    assert umdo_formulation._n_samples is None
    assert "n_samples" not in doe_algo_options

    # In the options of the DOE,
    # the number of samples is set to 3 with the property _n_samples.
    umdo_formulation._n_samples = 3
    assert umdo_formulation._n_samples == doe_algo_options["n_samples"] == 3


def test_save_samples(disciplines, design_space, uncertain_space, tmp_wd):
    scenario = UDOEScenario(
        disciplines,
        "MDF",
        "f",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation_parameters={
            "samples_directory_path": "foo",
            "algo": "CustomDOE",
            "algo_options": {"samples": array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])},
        },
    )
    scenario.add_constraint("c", "Margin", factor=3.0)
    scenario.add_observable("o", "Variance")
    scenario.execute({
        "algo": "CustomDOE",
        "algo_options": {
            "samples": array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        },
    })
    assert set(Path("foo").iterdir()) == {Path("foo") / "1.pkl", Path("foo") / "2.pkl"}

    expected_dataset = IODataset()
    expected_dataset.add_input_variable("u", array([[0.0], [1.0]]))
    expected_dataset.add_input_variable("u1", array([[0.0], [1.0]]))
    expected_dataset.add_input_variable("u2", array([[0.0], [1.0]]))
    expected_dataset.add_output_variable("c", array([[0.0], [-3.0]]))
    expected_dataset.add_output_variable("f", array([[0.0], [-4.0]]))
    expected_dataset.add_output_variable("o", array([[0.0], [-2.0]]))
    with (Path("foo") / "1.pkl").open("rb") as f:
        dataset = pickle.load(f)

    assert_frame_equal(dataset, expected_dataset)
    assert compare_dict_of_arrays(
        dataset.misc, {"x0": array([0.0]), "x1": array([0.0]), "x2": array([0.0])}
    )
    assert dataset.name == "Iteration 1"

    expected_dataset = IODataset()
    expected_dataset.add_input_variable("u", array([[0.0], [1.0]]))
    expected_dataset.add_input_variable("u1", array([[0.0], [1.0]]))
    expected_dataset.add_input_variable("u2", array([[0.0], [1.0]]))
    expected_dataset.add_output_variable("c", array([[-9.0], [-12.0]]))
    expected_dataset.add_output_variable("f", array([[-9.0], [-13.0]]))
    expected_dataset.add_output_variable("o", array([[-9.0], [-11.0]]))
    with (Path("foo") / "2.pkl").open("rb") as f:
        dataset = pickle.load(f)

    assert_frame_equal(dataset, expected_dataset)
    assert compare_dict_of_arrays(
        dataset.misc, {"x0": array([1.0]), "x1": array([1.0]), "x2": array([1.0])}
    )
    assert dataset.name == "Iteration 2"


def test_iterative_margin():
    """"""
