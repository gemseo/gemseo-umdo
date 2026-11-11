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

import sys
from importlib import import_module
from textwrap import dedent
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from unittest import mock

import pytest
from gemseo.core.function.array_function import ArrayFunction
from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import CustomDOE_Settings
from gemseo.formulation import MDF_Settings
from gemseo.mda import MDAChain_Settings
from gemseo.mda import MDAGaussSeidel_Settings
from gemseo.optimization import OptimizationProblem
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.util.testing.helper import assert_exception
from numpy import array
from numpy import array_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.formulation._estimator.sampling.factory import SamplingEstimatorFactory
from gemseo_umdo.formulation.core.base import BaseUMDOFormulation
from gemseo_umdo.formulation.sampling_settings import Sampling_Settings
from gemseo_umdo.formulation.taylor_polynomial_settings import TaylorPolynomial_Settings
from gemseo_umdo.scenario.mdo import UMDOScenario
from gemseo_umdo.statistic.margin_settings import Margin_Settings
from gemseo_umdo.statistic.quantile_settings import Quantile_Settings

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from types import ModuleType

    from gemseo_umdo.formulation.core.base_settings import BaseUMDOFormulationSettings
    from gemseo_umdo.statistic.core.base_settings import BaseStatisticSettings


@pytest.fixture
def disciplines() -> list[AnalyticDiscipline]:
    """Three coupled disciplines, with two strongly coupled ones."""
    disc0 = AnalyticDiscipline(
        {"f": "x0+y1+y2", "c": "x0+y1+y2", "o": "x0+y1+y2"}, name="D0"
    )
    disc1 = AnalyticDiscipline({"y1": "x0+x1+2*y2"}, name="D1")
    disc2 = AnalyticDiscipline({"y2": "x0+x2+y1+u"}, name="D2")
    return [disc0, disc1, disc2]


@pytest.fixture
def design_space() -> DesignSpace:
    """The design space containing the global and local design variables."""
    space = DesignSpace()
    for name in ["x0", "x1", "x2"]:
        space.add_variable(name, lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("y1", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return space


@pytest.fixture
def uncertain_space() -> RandomSpace:
    """The uncertain space containing the random variable."""
    space = RandomSpace()
    space.add_variable("u", OTNormalDistribution_Settings())
    return space


@pytest.fixture
def mdf_settings() -> MDF_Settings:
    """The settings of the MDF formulation."""
    return MDF_Settings(
        main_mda_settings=MDAChain_Settings(
            inner_mda_settings=MDAGaussSeidel_Settings()
        )
    )


class StatisticFunction(ArrayFunction):
    """A function to compute a statistic."""

    def __init__(  # noqa: D107
        self,
        formulation: BaseUMDOFormulation,
        output_name: str,
        function_type: str,
        statistic_settings: BaseStatisticSettings,
    ) -> None:
        super().__init__(lambda u: array([1.0]), name="func")
        self.mock = f"{output_name}_statistics"
        self.f_type = ArrayFunction.ConstraintType.INEQ
        statistic_estimator = mock.Mock()
        statistic_estimator.factor = 2
        self.statistic_estimator = statistic_estimator


class MyUMDOFormulation(BaseUMDOFormulation):
    """A dummy BaseUMDOFormulation."""

    settings_class: ClassVar[type[Sampling_Settings]] = Sampling_Settings

    def __init__(self, *args, **kwargs):  # noqa: D107
        self._statistic_function_class = StatisticFunction
        self._statistic_factory = SamplingEstimatorFactory()
        super().__init__(*args, **kwargs)


@pytest.fixture
def formulation(disciplines, design_space, mdf_settings, uncertain_space):
    """A dummy formulation with an observable and a constraint."""
    form = MyUMDOFormulation(
        OptimizationProblem(design_space),
        disciplines,
        Sampling_Settings(n_samples=10),
        uncertain_space=uncertain_space,
        mdo_formulation_settings=mdf_settings,
    )
    form.problem.objective = form.create_objective("f", "Mean")
    form.add_constraint("c", Margin_Settings(factor=3.0))
    form.add_observable("o", "Mean")
    return form


def test_uncertain_space(formulation):
    """Check that the uncertain space contains the uncertain variable."""
    assert list(formulation.uncertain_space.variables) == ["u"]


def test_name(formulation):
    """Check the name of the BaseUMDOFormulation."""
    assert formulation.name == "MyUMDOFormulation[MDF]"


def test_objective(formulation):
    """Check the objective function is correctly set."""
    assert formulation.problem.objective.mock == "f_statistics"
    assert formulation.problem.objective.name == "E[f]"


def test_observable(formulation):
    """Check the observable function is correctly set."""
    assert formulation.problem.observables[0].mock == "o_statistics"
    assert formulation.problem.observables[0].name == "E[o]"


def test_constraint(formulation):
    """Check the constraint function is correctly set."""
    opt_problem = formulation.problem
    assert opt_problem.constraints[0].mock == "c_statistics"
    assert opt_problem.constraints[0].name == "Margin[c; 3.0]"


def test_available_statistics(formulation):
    """Check the available statistics returned by a property."""
    assert formulation.available_statistics == [
        "Margin",
        "Mean",
        "Probability",
        "Quantile",
        "StandardDeviation",
        "Variance",
    ]


def test_update_top_level_disciplines(formulation):
    """Check the update of the top-level discipline."""
    formulation.update_top_level_disciplines(array([1, 2, 3]))
    for discipline in formulation.get_top_level_disciplines():
        assert discipline.default_input_data["x0"] == array([1])
        assert discipline.default_input_data["x1"] == array([2])
        assert discipline.default_input_data["x2"] == array([3])


def test_init_sub_formulation(formulation):
    """Check the properties of the MDO formulation."""
    sub_form = formulation.mdo_formulation
    assert sub_form.__class__.__name__ == "MDF"
    assert sub_form.mda.inner_mdas[0].name == "MDAGaussSeidel"
    assert sub_form.disciplines == formulation.disciplines
    assert [o.name for o in sub_form.problem.observables] == ["f", "c", "o"]
    assert list(sub_form.input_space.variables) == ["u"]


def test_multiobjective(disciplines, design_space, uncertain_space):
    """Check the name of the objective function for a multiobjective case."""
    formulation = MyUMDOFormulation(
        OptimizationProblem(design_space),
        disciplines,
        Sampling_Settings(n_samples=10),
        uncertain_space=uncertain_space,
    )
    formulation.problem.objective = formulation.create_objective(["f", "o"], "Mean")
    assert formulation.problem.objective.name == "E[f_o]"


def _execute_margin_scenario(
    disciplines: Sequence[AnalyticDiscipline],
    design_space: DesignSpace,
    uncertain_space: RandomSpace,
    statistic_estimation_settings: BaseUMDOFormulationSettings,
    parameters: Mapping[str, float],
    positive: bool = False,
    maximize: bool = False,
) -> dict[str, Any]:
    """Execute a scenario whose functions are margins.

    Args:
        disciplines: The disciplines.
        design_space: The design space.
        uncertain_space: The uncertain space.
        statistic_estimation_settings: The settings of the U-MDO formulation.
        parameters: The parameters of the margins.
        positive: Whether the inequality constraint is positive.
        maximize: Whether to maximize the objective.

    Returns:
        The last item of the database.
    """
    scenario = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
        statistic_estimation_settings=statistic_estimation_settings,
        formulation_settings=MDF_Settings(),
    )
    statistic_settings = Margin_Settings(**parameters)
    scenario.add_objective("f", statistic_settings, minimize=not maximize)
    scenario.add_constraint("c", statistic_settings, positive=positive)
    scenario.add_observable("o", statistic_settings)
    scenario.execute(CustomDOE_Settings(samples=array([[1.0, 1.0, 1.0]])))
    return scenario.formulation.problem.database.last_item


@pytest.mark.parametrize(
    ("positive", "constraint_name"),
    [(False, "Margin[c; 3.0]"), (True, "-Margin[c; -3.0]")],
    ids=["negative", "positive"],
)
@pytest.mark.parametrize(
    ("maximize", "objective_name"),
    [(False, "Margin[f; 3.0]"), (True, "-Margin[f; -3.0]")],
    ids=["minimize", "maximize"],
)
def test_margin_objective_and_constraint(
    disciplines,
    design_space,
    uncertain_space,
    statistic_estimation_settings,
    positive,
    constraint_name,
    maximize,
    objective_name,
):
    """Check that a margin objective or inequality constraint is conservative.

    The margin is conservative from below
    for an objective to maximize or a positive constraint;
    the conservative factors are checked by the tests of the settings.
    """
    args = (disciplines, design_space, uncertain_space, statistic_estimation_settings)
    last_item = _execute_margin_scenario(
        *args, {"factor": -3.0}, positive=positive, maximize=maximize
    )
    assert objective_name in last_item
    assert constraint_name in last_item


def test_margin_observable_default_factor(
    disciplines, design_space, uncertain_space, statistic_estimation_settings
):
    """Check that the default factor of a margin observable is 2."""
    args = (disciplines, design_space, uncertain_space, statistic_estimation_settings)
    reference = _execute_margin_scenario(*args, {"factor": 2.0})
    last_item = _execute_margin_scenario(*args, {})
    assert_equal(last_item["Margin[o; 2.0]"], reference["Margin[o; 2.0]"])


def test_margin_observable_negative_factor(
    disciplines, design_space, uncertain_space, statistic_estimation_settings
):
    """Check that the negative factor of a margin observable is used as is."""
    args = (disciplines, design_space, uncertain_space, statistic_estimation_settings)
    reference = _execute_margin_scenario(*args, {"factor": 3.0})
    last_item = _execute_margin_scenario(*args, {"factor": -3.0})
    assert not array_equal(last_item["Margin[o; -3.0]"], reference["Margin[o; 3.0]"])


@pytest.mark.parametrize(
    ("parameters", "name"),
    [({}, "Margin[c; 2.0]"), ({"factor": -3.0}, "Margin[c; -3.0]")],
    ids=["default", "negative"],
)
@pytest.mark.parametrize("positive", [False, True])
def test_margin_equality_constraint(
    disciplines, design_space, uncertain_space, parameters, name, positive
):
    """Check that the factor of a margin equality constraint is used as is."""
    scenario = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
        statistic_estimation_settings=Sampling_Settings(n_samples=10),
        formulation_settings=MDF_Settings(),
    )
    scenario.add_constraint(
        "c",
        Margin_Settings(**parameters),
        constraint_type=ArrayFunction.ConstraintType.EQ,
        positive=positive,
    )
    assert scenario.formulation.problem.constraints[-1].name.endswith(name)


def test_unavailable_statistic(formulation, snapshot):
    """Check that an unavailable statistic raises an explicit error."""
    with assert_exception(ValueError, snapshot):
        formulation.add_observable("o", "Foo")


def test_unsupported_statistic(disciplines, design_space, uncertain_space, snapshot):
    """Check the error raised for a statistic not supported by a formulation."""
    scenario = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
        statistic_estimation_settings=TaylorPolynomial_Settings(),
        formulation_settings=MDF_Settings(),
    )
    with assert_exception(ValueError, snapshot):
        scenario.add_objective("f", Quantile_Settings())


def _set_factor(settings: Margin_Settings) -> None:
    """Change the factor of the settings of a margin.

    Args:
        settings: The settings of the margin.
    """
    settings.factor = 5.0


def _append_offset(settings: BaseStatisticSettings) -> None:
    """Append an offset to the mutable field of the settings of the user.

    Args:
        settings: The settings of the statistic of the user.
    """
    settings.offsets.append(1.0)


@pytest.mark.parametrize(
    ("is_margin", "is_objective", "change_settings", "function_name"),
    [
        (True, False, _set_factor, "Margin[f; 3.0]"),
        (False, False, _append_offset, "scaled_mean[f; 3.0]"),
        (False, True, _append_offset, "scaled_mean[f; 3.0]"),
    ],
    ids=["margin-observable", "mutable-field-observable", "mutable-field-objective"],
)
def test_settings_changed_after_use(
    disciplines,
    design_space,
    uncertain_space,
    user_statistic_module,
    is_margin,
    is_objective,
    change_settings,
    function_name,
):
    """Check that changing the settings after adding a statistic has no effect.

    The settings are deep-copied,
    even when an objective makes them conservative with a shallow copy.
    """

    def execute(change):
        scenario = UMDOScenario(
            disciplines,
            design_space,
            uncertain_space,
            statistic_estimation_settings=Sampling_Settings(
                n_samples=10, estimate_statistics_iteratively=False
            ),
            formulation_settings=MDF_Settings(),
        )
        settings = (
            Margin_Settings(factor=3.0)
            if is_margin
            else user_statistic_module.ScaledMean_Settings(scale=3.0)
        )
        if is_objective:
            scenario.add_objective("f", settings)
        else:
            scenario.add_objective("f", "Mean")
            scenario.add_observable("f", settings)

        if change:
            change_settings(settings)

        scenario.execute(CustomDOE_Settings(samples=array([[1.0, 1.0, 1.0]])))
        return scenario.formulation.problem.database.last_item[function_name]

    assert_almost_equal(execute(True), execute(False))


_user_statistic_module_name: Final[str] = "user_statistic_for_gemseo_umdo"
"""The name of the module defining a statistic of the user."""

_user_statistic_module = '''
from __future__ import annotations

from pydantic import Field

from gemseo_umdo.statistic.core.base_settings import BaseStatisticSettings
from gemseo_umdo.formulation._estimator.sampling.base import (
    BaseSamplingEstimator,
)


class ScaledMean_Settings(BaseStatisticSettings):
    """The settings of a shifted and scaled mean."""

    scale: float = 1.0

    offsets: list[float] = Field(default_factory=list)

    def make_conservative(self, from_below):
        # A shallow copy sharing the mutable fields with these settings.
        return self.model_copy(update={"scale": self.scale})


# The base class is not parametrized by the settings class,
# as the cache of typing would keep the settings class alive after the test,
# and so registered as a subclass of BaseStatisticSettings.
class ScaledMean(BaseSamplingEstimator):
    """Estimator of a shifted and scaled mean."""

    def estimate_statistic(self, samples):
        return self._settings.scale * samples.mean(0) + sum(self._settings.offsets)

    def compute_jacobian(self, samples, jac_samples):
        return self._settings.scale * jac_samples.mean(0)
'''
"""A module defining a statistic of the user."""


@pytest.fixture
def user_statistic_module(tmp_wd, monkeypatch, reset_factory) -> ModuleType:
    """A module of the user defining a statistic, found through GEMSEO_PATH.

    The module is removed from the imported modules after the test.
    """
    (tmp_wd / f"{_user_statistic_module_name}.py").write_text(
        dedent(_user_statistic_module)
    )
    monkeypatch.setenv("GEMSEO_PATH", str(tmp_wd))
    monkeypatch.syspath_prepend(str(tmp_wd))
    yield import_module(_user_statistic_module_name)
    del sys.modules[_user_statistic_module_name]


def test_user_statistic(
    disciplines, design_space, uncertain_space, user_statistic_module
):
    """Check that a user can add a statistic with its settings and estimator."""

    def execute(statistic_settings):
        scenario = UMDOScenario(
            disciplines,
            design_space,
            uncertain_space,
            statistic_estimation_settings=Sampling_Settings(
                n_samples=10, estimate_statistics_iteratively=False
            ),
            formulation_settings=MDF_Settings(),
        )
        scenario.add_objective("f", statistic_settings)
        scenario.execute(CustomDOE_Settings(samples=array([[1.0, 1.0, 1.0]])))
        return scenario.formulation.problem

    problem = execute(user_statistic_module.ScaledMean_Settings(scale=3.0))
    reference_problem = execute("Mean")
    assert problem.objective.name == "scaled_mean[f; 3.0]"
    assert_almost_equal(
        problem.database.last_item[problem.objective.name],
        3.0 * reference_problem.database.last_item["E[f]"],
    )
