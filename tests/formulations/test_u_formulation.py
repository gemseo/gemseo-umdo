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

from typing import Any

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.mdf import MDF
from numpy import array

from gemseo_umdo.formulations._statistics.sampling.factory import (
    SamplingEstimatorFactory,
)
from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation
from gemseo_umdo.formulations.sampling_settings import SamplingSettings


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
def uncertain_space() -> ParameterSpace:
    """The uncertain space containing the random variable."""
    space = ParameterSpace()
    space.add_random_variable("u", "SPNormalDistribution")
    return space


@pytest.fixture
def mdf(disciplines, uncertain_space) -> MDF:
    """The MDF formulation."""
    return MDF(
        disciplines,
        "f",
        uncertain_space,
        main_mda_settings={"inner_mda_name": "MDAGaussSeidel"},
    )


class StatisticFunction(MDOFunction):
    """A function to compute a statistic."""

    def __init__(  # noqa: D107
        self,
        formulation: BaseUMDOFormulation,
        func: MDOFunction,
        function_type: str,
        name: str,
        **parameters: Any,
    ) -> None:
        super().__init__(lambda u: array([1.0]), name="func")
        self.mock = f"{func.name}_statistics"
        self.f_type = func.ConstraintType.INEQ


class MyUMDOFormulation(BaseUMDOFormulation):
    """A dummy BaseUMDOFormulation."""

    Settings = SamplingSettings

    def __init__(self, *args, **kwargs):  # noqa: D107
        self._statistic_function_class = StatisticFunction
        self._statistic_factory = SamplingEstimatorFactory()
        super().__init__(*args, **kwargs)


@pytest.fixture
def formulation(disciplines, design_space, mdf, uncertain_space):
    """A dummy formulation with an observable and a constraint."""
    form = MyUMDOFormulation(
        disciplines,
        "f",
        design_space,
        mdf,
        uncertain_space,
        "Mean",
    )
    form.add_constraint("c", "Margin", factor=3.0)
    form.add_observable("o", "Mean")
    return form


def test_uncertain_space(formulation):
    """Check that the uncertain space contains the uncertain variable."""
    assert formulation.uncertain_space.variable_names == ["u"]


def test_name(formulation):
    """Check the name of the BaseUMDOFormulation."""
    assert formulation.name == "MyUMDOFormulation[MDF]"


def test_objective(formulation):
    """Check the objective function is correctly set."""
    assert formulation.optimization_problem.objective.mock == "f_statistics"
    assert formulation.optimization_problem.objective.name == "E[f]"


def test_observable(formulation):
    """Check the observable function is correctly set."""
    assert formulation.optimization_problem.observables[0].mock == "o_statistics"
    assert formulation.optimization_problem.observables[0].name == "E[o]"


def test_constraint(formulation):
    """Check the constraint function is correctly set."""
    opt_problem = formulation.optimization_problem
    assert opt_problem.constraints[0].mock == "c_statistics"
    assert opt_problem.constraints[0].name == "Margin[c; 3.0]"


def test_available_statistics(formulation):
    """Check the available statistics returned by a property."""
    assert formulation.available_statistics == [
        "Margin",
        "Mean",
        "Probability",
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
    assert sub_form.optimization_problem.objective.name == "f"
    assert sub_form.optimization_problem.observables[0].name == "c"
    assert sub_form.optimization_problem.observables[1].name == "o"
    assert sub_form.design_space.variable_names == ["u"]


def test_multiobjective(disciplines, design_space, mdf, uncertain_space):
    """Check the name of the objective function for a multiobjective case."""
    formulation = MyUMDOFormulation(
        disciplines,
        ["f", "o"],
        design_space,
        mdf,
        uncertain_space,
        "Mean",
    )
    assert formulation.optimization_problem.objective.name == "E[f_o]"
