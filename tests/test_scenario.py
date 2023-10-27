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

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo_umdo.formulations.factory import UMDOFormulationsFactory
from gemseo_umdo.formulations.sampling import Sampling
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

AVAILABLE_FORMULATIONS = UMDOFormulationsFactory().formulations


@pytest.fixture
def disciplines() -> list[MDODiscipline]:
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
        space.add_variable(name, l_b=0.0, u_b=1.0, value=0.5)
    space.add_variable("y1", l_b=0.0, u_b=1.0, value=0.5)
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
        "MDF",
        "f",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation="Sampling",
        statistic_estimation_parameters={"algo": "OT_OPT_LHS", "n_samples": 3},
        inner_mda_name="MDAGaussSeidel",
    )
    scn.add_constraint("c", "Margin", factor=3.0)
    scn.add_observable("o", "Mean")
    return scn


def test_disciplines_copy(scenario, disciplines):
    """Check that the _UScenario works on a copy of the sequence of disciplines."""
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
    """Check the text representation of the formulation section of the scenario."""
    expected = f"   Formulation: {scenario.formulation.name}"
    assert repr(scenario).split("\n")[2] == expected


def test_mdo_formulation(scenario):
    """Check the content of the MDO formulation."""
    mdo_formulation = scenario.mdo_formulation
    opt_problem = mdo_formulation.opt_problem
    assert mdo_formulation.__class__.__name__ == "MDF"
    assert mdo_formulation.mda.inner_mdas[0].name == "MDAGaussSeidel"
    assert mdo_formulation.disciplines == scenario.disciplines
    assert opt_problem.objective.name == "f"
    assert [c.name for c in opt_problem.constraints] == ["c"]
    assert [o.name for o in opt_problem.observables] == [
        "Mean[f]",
        "Margin[c]",
        "o",
        "Mean[o]",
    ]
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
        "MDF",
        "f",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation="Sampling",
        statistic_estimation_parameters={"algo": "OT_OPT_LHS", "n_samples": 3},
        **kwargs,
    )
    maximize = bool(maximize_objective)
    assert scn.formulation.mdo_formulation.opt_problem.minimize_objective
    assert scn.formulation.opt_problem.minimize_objective is not maximize
    expected_name = "-E[f]" if maximize else "E[f]"
    assert scn.formulation.opt_problem.objective.name == expected_name


def test_uncertain_design_variables(disciplines, design_space, uncertain_space):
    """Check that a design variable can be noised."""
    scn = UMDOScenario(
        disciplines,
        "MDF",
        "f",
        design_space,
        uncertain_space,
        "Mean",
        uncertain_design_variables={"x0": "{}+v"},
        statistic_estimation="Sampling",
        statistic_estimation_parameters={"algo": "OT_OPT_LHS", "n_samples": 3},
    )
    design_space = scn.design_space
    assert "x0" not in design_space
    assert "dv_x0" in design_space
    discipline = scn.mdo_formulation.disciplines[-1]
    assert discipline.name == "Design Uncertainties"
    assert discipline.expressions == {"x0": "dv_x0+v"}
    assert len(scn.disciplines) == len(disciplines) + 1
    for index, discipline in enumerate(disciplines):
        assert id(scn.disciplines[index]) == id(discipline)


def test_statistic_no_estimation_parameters(disciplines, design_space, uncertain_space):
    """Check that a TypeError is raised when estimation parameters are missing.

    The default UMDOFormulation is "Sampling" whose "n_samples" argument is mandatory.
    """
    with pytest.raises(
        TypeError,
        match=re.escape(
            "__init__() missing 1 required positional argument: 'n_samples'"
        ),
    ):
        UMDOScenario(
            disciplines,
            "MDF",
            "f",
            design_space,
            uncertain_space,
            "Mean",
            maximize_objective=True,
        )
