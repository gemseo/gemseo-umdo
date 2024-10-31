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

import re
from typing import TYPE_CHECKING

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.chains.chain import MDOChain
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.auto_py import AutoPyDiscipline
from numpy import array
from numpy import atleast_2d
from numpy import vstack
from numpy.linalg import norm

from gemseo_umdo.disciplines.additive_noiser import AdditiveNoiser
from gemseo_umdo.disciplines.multiplicative_noiser import MultiplicativeNoiser
from gemseo_umdo.formulations.factory import UMDOFormulationsFactory
from gemseo_umdo.formulations.sampling import Sampling
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline import Discipline

AVAILABLE_FORMULATIONS = UMDOFormulationsFactory().class_names


@pytest.fixture
def disciplines() -> list[Discipline]:
    """Three simple disciplines."""
    disc0 = AnalyticDiscipline(
        {"f": "x0+y1+y2", "c": "x0+y1+y2", "o": "x0+y1+y2"}, name="D0"
    )
    disc1 = AnalyticDiscipline({"y1": "x0+x1+2*y2"}, name="D1")
    disc2 = AnalyticDiscipline({"y2": "x0+x2+y1+u"}, name="D2")
    return [disc0, disc1, disc2]


@pytest.fixture
def design_space() -> DesignSpace:
    """The space of local and global design variables."""
    space = DesignSpace()
    for name in ["x0", "x1", "x2"]:
        space.add_variable(name, lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("y1", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return space


@pytest.fixture
def uncertain_space() -> ParameterSpace:
    """The space defining the uncertain variable."""
    space = ParameterSpace()
    space.add_random_variable("u", "SPNormalDistribution")
    return space


@pytest.fixture
def scenario(disciplines, design_space, uncertain_space) -> UMDOScenario:
    """The MDO scenario under uncertainty."""
    scn = UMDOScenario(
        disciplines,
        "f",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="MDF",
        statistic_estimation="Sampling",
        statistic_estimation_parameters={"algo": "OT_OPT_LHS", "n_samples": 3},
        main_mda_settings={"inner_mda_name": "MDAGaussSeidel"},
    )
    scn.add_constraint("c", "Margin", factor=3.0)
    scn.add_observable("o", "Mean")
    return scn


def test_disciplines_copy(scenario, disciplines):
    """Check that the BaseUScenario works on a copy of the sequence of disciplines."""
    assert id(scenario.disciplines) != id(disciplines)


def test_available_statistics(scenario):
    """Check the property returning the names of the available statistics."""
    expected = set(scenario.formulation.available_statistics)
    assert set(scenario.available_statistics) == expected


def test_factory(scenario):
    """Check that the UMDOScenario uses a UMDOFormulationsFactory."""
    assert isinstance(scenario._formulation_factory, UMDOFormulationsFactory)


def test_formulation(scenario):
    """Check the formulation after instantiation."""
    assert isinstance(scenario.formulation, Sampling)
    assert scenario.formulation_name == "Sampling[MDF; OT_OPT_LHS(3)]"
    assert scenario.formulation._algo.algo_name
    assert scenario.formulation._n_samples == 3


def test_design_space(scenario):
    """Check that the design space contains the design variables."""
    assert set(scenario.design_space.variable_names) == {"x0", "x1", "x2"}


def test_uncertain_space(scenario):
    """Check that the uncertain space contains the uncertain variables."""
    assert set(scenario.uncertain_space.variable_names) == {"u"}


def test_repr(scenario):
    """Check the string representation of the scenario."""
    expected = """UMDOScenario
   Disciplines: D0 D1 D2
   Formulation:
      MDO formulation: MDF
      Statistic estimation: Sampling
   Uncertain space:
      +------+-------------------------+
      | Name |       Distribution      |
      +------+-------------------------+
      |  u   | norm(mu=0.0, sigma=1.0) |
      +------+-------------------------+"""
    assert repr(scenario) == expected


def test_mdo_formulation(scenario):
    """Check the content of the MDO formulation."""
    mdo_formulation = scenario.mdo_formulation
    opt_problem = mdo_formulation.optimization_problem
    assert mdo_formulation.__class__.__name__ == "MDF"
    assert mdo_formulation.mda.inner_mdas[0].name == "MDAGaussSeidel"
    assert mdo_formulation.disciplines == scenario.disciplines
    assert opt_problem.objective.name == "f"
    assert [o.name for o in opt_problem.observables] == ["c", "o"]
    assert scenario.mdo_formulation.design_space.variable_names == ["u"]


@pytest.mark.parametrize("maximize_objective", [None, False, True])
def test_maximize_objective(
    disciplines, design_space, uncertain_space, maximize_objective
):
    """Check that the argument maximize_objective is correctly used."""
    if maximize_objective is None:
        kwargs = {}
    else:
        kwargs = {"maximize_objective": maximize_objective}
    scn = UMDOScenario(
        disciplines,
        "f",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="MDF",
        statistic_estimation="Sampling",
        statistic_estimation_parameters={"algo": "OT_OPT_LHS", "n_samples": 3},
        **kwargs,
    )
    maximize = bool(maximize_objective)
    assert scn.formulation.mdo_formulation.optimization_problem.minimize_objective
    assert scn.formulation.optimization_problem.minimize_objective is not maximize
    expected_name = "-E[f]" if maximize else "E[f]"
    assert scn.formulation.optimization_problem.objective.name == expected_name


def test_uncertain_design_variables(disciplines, design_space, uncertain_space):
    """Check that a design variable can be noised.

    Here we check the disciplines created by the BaseUScenario.
    """
    scn = UMDOScenario(
        disciplines,
        "f",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="MDF",
        uncertain_design_variables={
            "x0": ("+", "v0"),
            "x1": "{}+v1",
            "x2": ("*", "v2"),
        },
        statistic_estimation="Sampling",
        statistic_estimation_parameters={"algo": "OT_OPT_LHS", "n_samples": 3},
    )
    design_space = scn.design_space
    for i in range(3):
        assert f"x{i}" not in design_space
        assert f"dv_x{i}" in design_space

    assert len(scn.disciplines) == len(disciplines) + 1
    mdo_chain = scn.disciplines[0]
    assert isinstance(mdo_chain, MDOChain)

    discipline = mdo_chain.disciplines[0]
    assert isinstance(discipline, AnalyticDiscipline)
    assert discipline.expressions == {"x1": "dv_x1+v1"}

    discipline = mdo_chain.disciplines[1]
    assert isinstance(discipline, AdditiveNoiser)
    assert set(discipline.input_grammar.names) == {"dv_x0", "v0"}
    assert set(discipline.output_grammar.names) == {"x0"}

    discipline = mdo_chain.disciplines[2]
    assert isinstance(discipline, MultiplicativeNoiser)
    assert set(discipline.input_grammar.names) == {"dv_x2", "v2"}
    assert set(discipline.output_grammar.names) == {"x2"}


@pytest.mark.parametrize("x", [array([1, 2]), array([1])])
@pytest.mark.parametrize(
    ("u1", "u2"),
    [(array([1.0]), array([-1.0])), (array([1.0, -1.0]), array([-1.0, 1.0]))],
)
def test_uncertain_design_variables_values(x, u1, u2):
    """Check that a design variable can be noised.

    Here we check the disciplines.
    """
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_vector("u", "OTNormalDistribution", size=len(u1))

    def f(x):
        y = norm(x) ** 2
        return y  # noqa: RET504

    design_space = DesignSpace()
    design_space.add_variable("x", size=len(x), value=0)

    discipline = AutoPyDiscipline(f)
    scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="DisciplinaryOpt",
        statistic_estimation_parameters={
            "algo": "CustomDOE",
            "algo_options": {"samples": vstack((u1, u2))},
        },
        uncertain_design_variables={"x": ("+", "u")},
    )
    scenario.execute(algo_name="CustomDOE", samples=atleast_2d(x))
    assert scenario.optimization_result.f_opt == (f(x + u1) + f(x + u2)) / 2


def test_statistic_no_estimation_parameters(disciplines, design_space, uncertain_space):
    """Check that a TypeError is raised when estimation parameters are missing.

    The default BaseUMDOFormulation is "Sampling" whose "n_samples" argument is
    mandatory for most of the DOE algorithms, including the default one.
    """
    with pytest.raises(ValueError, match=re.escape("Sampling: n_samples is required.")):
        UMDOScenario(
            disciplines,
            "f",
            design_space,
            uncertain_space,
            "Mean",
            formulation_name="MDF",
            maximize_objective=True,
        )


@pytest.mark.parametrize(
    ("constraint_name", "constraint_expr", "constraint_res"),
    [
        (None, "E[y] <= 0.05", "[E[y]-0.05] = [1.45]"),
        ("foo", "foo: E[y] <= 0.05", "foo = [1.45]"),
    ],
)
@pytest.mark.parametrize(
    ("maximize_objective", "use_standardized_objective", "objective_expr"),
    [
        (False, False, "minimize E[y]"),
        (False, True, "minimize E[y]"),
        (True, False, "maximize E[y]"),
        (True, True, "minimize -E[y]"),
    ],
)
def test_log(
    caplog,
    constraint_name,
    constraint_expr,
    constraint_res,
    maximize_objective,
    use_standardized_objective,
    objective_expr,
):
    """Check some parts of the log of a scenario."""
    discipline = AnalyticDiscipline({"y": "x**2+u"}, name="f")

    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)

    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("u", "OTNormalDistribution")

    scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="DisciplinaryOpt",
        statistic_estimation="Sampling",
        statistic_estimation_parameters={
            "algo": "CustomDOE",
            "n_samples": None,
            "algo_options": {"samples": array([[0.5]])},
        },
        maximize_objective=maximize_objective,
    )

    scenario.add_constraint(
        "y",
        "Mean",
        constraint_type="ineq",
        value=0.05,
        constraint_name=constraint_name,
    )
    scenario.use_standardized_objective = use_standardized_objective
    scenario.execute(algo_name="CustomDOE", samples=array([[1.0]]))
    assert objective_expr in caplog.text
    assert constraint_expr in caplog.text
    assert constraint_res in caplog.text
