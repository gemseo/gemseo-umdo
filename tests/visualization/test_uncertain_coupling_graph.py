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

import warnings
from typing import TYPE_CHECKING

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import CustomDOE_Settings
from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.problem.mdo.sobieski import SobieskiAerodynamics
from gemseo.problem.mdo.sobieski import SobieskiMission
from gemseo.problem.mdo.sobieski import SobieskiProblem
from gemseo.problem.mdo.sobieski import SobieskiPropulsion
from gemseo.problem.mdo.sobieski import SobieskiStructure
from gemseo.space import DesignSpace
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.testing.helper import assert_exception
from graphviz import Digraph
from numpy import array

from gemseo_umdo.visualization.uncertain_coupling_graph import UncertainCouplingGraph

if TYPE_CHECKING:
    from gemseo.problem.mdo.sobieski.discipline import SobieskiDiscipline


@pytest.fixture(scope="module")
def uncertain_space() -> DesignSpace:
    """The uncertain space of the Sobieski's SSBJ problem."""
    design_space = SobieskiProblem().design_space
    design_variable_names = ["x_1", "x_2", "x_3", "x_shared"]
    design_space.filter(design_variable_names)
    optimum_design = split_array_to_dict_of_arrays(
        SobieskiProblem().optimum_design,
        {n: v.size for n, v in design_space.variables.items()},
        design_variable_names,
    )
    space = DesignSpace()
    for name, value in optimum_design.items():
        space.add_variable(
            name,
            size=value.size,
            lower_bound=value * 0.95,
            upper_bound=value * 1.05,
            value=value,
        )
    return space


@pytest.fixture(scope="module")
def disciplines() -> list[SobieskiDiscipline]:
    """The disciplines of the Sobieski's SSBJ problem."""
    return [
        SobieskiAerodynamics(),
        SobieskiStructure(),
        SobieskiPropulsion(),
        SobieskiMission(),
    ]


@pytest.fixture(scope="module")
def uncertain_coupling_graph(disciplines, uncertain_space) -> UncertainCouplingGraph:
    """The uncertain coupling graph of the Sobieski's SSBJ problem."""
    graph = UncertainCouplingGraph(disciplines, uncertain_space)
    graph.sample(OT_OPT_LHS_Settings(n_samples=10))
    return graph


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"maximum_thickness": 10},
        {"dispersion_measure": UncertainCouplingGraph.DispersionMeasure.CV},
        {"variable_names": ["y_21"]},
    ],
    ids=["default", "maximum_thickness", "dispersion_measure", "filter_names"],
)
def test_draw(uncertain_coupling_graph, kwargs, snapshot):
    """Check the graph drawn by draw()."""
    digraph = uncertain_coupling_graph.draw(**kwargs)
    assert isinstance(digraph, Digraph)
    assert digraph.source == snapshot


def test_no_coupling_names(uncertain_coupling_graph):
    """Check the graph without coupling variables of interest."""
    digraph = uncertain_coupling_graph.draw(variable_names=[])
    assert "->" not in digraph.source


def test_unknown_variable_names(disciplines, uncertain_space, snapshot):
    """Check that draw() rejects variables not defined at instantiation."""
    uncertain_coupling_graph = UncertainCouplingGraph(
        disciplines, uncertain_space, variable_names=["y_21"]
    )
    with assert_exception(ValueError, snapshot):
        uncertain_coupling_graph.draw(variable_names=["y_21", "y_12"])


@pytest.mark.parametrize(
    "dispersion_measure", list(UncertainCouplingGraph.DispersionMeasure)
)
def test_non_finite_dispersion(dispersion_measure, snapshot):
    """Check the thickness of the edges for a non-finite dispersion.

    The coupling variable y has a zero mean and a non-zero standard deviation,
    so its infinite dispersion is drawn with the maximum thickness,
    while w is constant equal to zero,
    so its undefined dispersion is drawn with zero thickness.
    The finite dispersion of x is greater than 1 and the largest one,
    so it is also drawn with the maximum thickness,
    and the one of v is drawn proportionally.
    """
    disciplines = [
        AnalyticDiscipline({"y": "u", "x": "u+0.25", "v": "u+4"}, name="D1"),
        AnalyticDiscipline({"w": "0*u"}, name="D2"),
        AnalyticDiscipline({"z": "y+w+x+v"}, name="D3"),
    ]
    uncertain_space = DesignSpace()
    uncertain_space.add_variable("u", lower_bound=-1.0, upper_bound=1.0)
    uncertain_coupling_graph = UncertainCouplingGraph(disciplines, uncertain_space)
    uncertain_coupling_graph.sample(CustomDOE_Settings(samples=array([[-1.0], [1.0]])))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        digraph = uncertain_coupling_graph.draw(dispersion_measure=dispersion_measure)

    assert digraph.source == snapshot


@pytest.mark.parametrize(
    "dispersion_measure", list(UncertainCouplingGraph.DispersionMeasure)
)
def test_no_positive_finite_dispersion(dispersion_measure, snapshot):
    """Check the thickness of the edges when no dispersion is positive and finite.

    The infinite dispersion of y is drawn with the maximum thickness
    and the undefined dispersion of w with zero thickness,
    although there is no finite dispersion to scale them.
    """
    disciplines = [
        AnalyticDiscipline({"y": "u"}, name="D1"),
        AnalyticDiscipline({"w": "0*u"}, name="D2"),
        AnalyticDiscipline({"z": "y+w"}, name="D3"),
    ]
    uncertain_space = DesignSpace()
    uncertain_space.add_variable("u", lower_bound=-1.0, upper_bound=1.0)
    uncertain_coupling_graph = UncertainCouplingGraph(disciplines, uncertain_space)
    uncertain_coupling_graph.sample(CustomDOE_Settings(samples=array([[-1.0], [1.0]])))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        digraph = uncertain_coupling_graph.draw(
            maximum_thickness=10, dispersion_measure=dispersion_measure
        )

    assert digraph.source == snapshot


@pytest.mark.parametrize(
    "variable_names",
    [["y_21"], (name for name in ["y_21"])],
    ids=["list", "generator"],
)
def test_output_names(disciplines, uncertain_space, variable_names, snapshot):
    """Check the graph drawn with custom output names set at instantiation."""
    uncertain_coupling_graph = UncertainCouplingGraph(
        disciplines, uncertain_space, variable_names=variable_names
    )
    uncertain_coupling_graph.sample(OT_OPT_LHS_Settings(n_samples=10))
    assert uncertain_coupling_graph.draw().source == snapshot


def test_repeated_sampling(disciplines, uncertain_space, snapshot):
    """Check that the dispersion is estimated from all the samples."""
    uncertain_coupling_graph = UncertainCouplingGraph(
        disciplines, uncertain_space, variable_names=["y_21"]
    )
    uncertain_coupling_graph.sample(OT_OPT_LHS_Settings(n_samples=10))
    uncertain_coupling_graph.sample(OT_OPT_LHS_Settings(n_samples=7, seed=3))
    assert uncertain_coupling_graph.draw().source == snapshot


def test_self_coupled(snapshot):
    """Check the graph drawn with a self-coupled discipline."""
    disciplines = [
        AnalyticDiscipline({"x": "(x+u)/2", "y": "u"}, name="D1"),
        AnalyticDiscipline({"z": "y"}, name="D2"),
    ]
    uncertain_space = DesignSpace()
    uncertain_space.add_variable("u", lower_bound=0.0, upper_bound=1)
    uncertain_coupling_graph = UncertainCouplingGraph(disciplines, uncertain_space)
    uncertain_coupling_graph.sample(OT_OPT_LHS_Settings(n_samples=10))
    assert uncertain_coupling_graph.draw(maximum_thickness=0.1).source == snapshot


def test_several_couplings_per_edge(snapshot):
    """Check that the couplings of an edge are drawn in alphabetical order."""
    disciplines = [
        AnalyticDiscipline({"c": "u", "a": "2*u", "b": "3*u"}, name="D1"),
        AnalyticDiscipline({"z": "a+b+c"}, name="D2"),
    ]
    uncertain_space = DesignSpace()
    uncertain_space.add_variable("u", lower_bound=1.0, upper_bound=3.0)
    uncertain_coupling_graph = UncertainCouplingGraph(disciplines, uncertain_space)
    uncertain_coupling_graph.sample(
        CustomDOE_Settings(samples=array([[1.0], [2.0], [3.0]]))
    )
    assert uncertain_coupling_graph.draw().source == snapshot
