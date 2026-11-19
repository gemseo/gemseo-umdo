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
import warnings
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import CustomDOE_Settings
from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.post.graph_view import GraphView
from gemseo.problem.mdo.sobieski import SobieskiAerodynamics
from gemseo.problem.mdo.sobieski import SobieskiMission
from gemseo.problem.mdo.sobieski import SobieskiProblem
from gemseo.problem.mdo.sobieski import SobieskiPropulsion
from gemseo.problem.mdo.sobieski import SobieskiStructure
from gemseo.space import DesignSpace
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.testing.helper import assert_exception
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


def check_dot_file(file_name: str) -> None:
    """Check that the content of a file.

    Compare the file located in the current wording directory
    with the file located in the directory of the current module.

    Args:
        file_name: The name of the file.
    """
    file_path = Path(file_name)
    file_path = file_path.with_suffix(".dot")
    assert (
        file_path.read_text().strip()
        == (Path(__file__).parent / "uncertain_coupling_graph" / file_path)
        .read_text()
        .strip()
    )


def test_default(uncertain_coupling_graph, tmp_wd):
    """Check the image computed with the default configuration."""
    file_name = "default.png"
    uncertain_coupling_graph.visualize(show=False, clean_up=False, file_path=file_name)
    check_dot_file(file_name)


def test_maximum_thickness(uncertain_coupling_graph, tmp_wd):
    """Check the image computed with a custom maximum thickness."""
    file_name = "maximum_thickness.png"
    uncertain_coupling_graph.visualize(
        show=False, clean_up=False, file_path=file_name, maximum_thickness=10
    )
    check_dot_file(file_name)


def test_dispersion_measure(uncertain_coupling_graph, tmp_wd):
    """Check the image computed with a custom dispersion measure."""
    file_name = "dispersion_measure.png"
    uncertain_coupling_graph.visualize(
        show=False,
        clean_up=False,
        file_path=file_name,
        dispersion_measure=UncertainCouplingGraph.DispersionMeasure.CV,
    )
    check_dot_file(file_name)


def test_filter_names(uncertain_coupling_graph, tmp_wd):
    """Check the image computed with custom names."""
    file_name = "filter_names.png"
    uncertain_coupling_graph.visualize(
        show=False,
        clean_up=False,
        file_path=file_name,
        variable_names=["y_21"],
    )
    check_dot_file(file_name)


def test_no_coupling_names(uncertain_coupling_graph):
    """Check the graph without coupling variables of interest."""
    graph_view = uncertain_coupling_graph.visualize(variable_names=[], save=False)
    assert "->" not in graph_view.source


def test_unknown_variable_names(disciplines, uncertain_space, snapshot):
    """Check that visualize() rejects variables not defined at instantiation."""
    uncertain_coupling_graph = UncertainCouplingGraph(
        disciplines, uncertain_space, variable_names=["y_21"]
    )
    with assert_exception(ValueError, snapshot):
        uncertain_coupling_graph.visualize(variable_names=["y_21", "y_12"], save=False)


@pytest.mark.parametrize(
    "dispersion_measure", list(UncertainCouplingGraph.DispersionMeasure)
)
def test_non_finite_dispersion(dispersion_measure, tmp_wd):
    """Check the thickness of the edges for a non-finite dispersion.

    The coupling variable y has a zero mean and a non-zero standard deviation,
    so its infinite dispersion is drawn with the maximum thickness,
    while w is constant equal to zero,
    so its undefined dispersion is drawn with zero thickness.
    The finite dispersion of x is greater than 1 and the largest one,
    so it is also drawn with the maximum thickness,
    and the one of v is drawn proportionally.
    """
    file_name = f"non_finite_dispersion_{dispersion_measure}.png"
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
        uncertain_coupling_graph.visualize(
            show=False,
            clean_up=False,
            file_path=file_name,
            dispersion_measure=dispersion_measure,
        )
    check_dot_file(file_name)


@pytest.mark.parametrize(
    "dispersion_measure", list(UncertainCouplingGraph.DispersionMeasure)
)
def test_no_positive_finite_dispersion(dispersion_measure):
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
        graph_view = uncertain_coupling_graph.visualize(
            save=False, maximum_thickness=10, dispersion_measure=dispersion_measure
        )

    source = graph_view.source
    assert (
        "D1 -> D3 [label=y color=black dir=forward fontcolor=black penwidth=10.0]"
        in source
    )
    assert (
        "D2 -> D3 [label=w color=black dir=forward fontcolor=black penwidth=0.0]"
        in source
    )


@pytest.mark.parametrize(
    "variable_names",
    [["y_21"], (name for name in ["y_21"])],
    ids=["list", "generator"],
)
def test_output_names(disciplines, uncertain_space, variable_names, tmp_wd):
    """Check the image computed with custom output names set at instantiation."""
    file_name = "output_names.png"
    uncertain_coupling_graph = UncertainCouplingGraph(
        disciplines, uncertain_space, variable_names=variable_names
    )
    uncertain_coupling_graph.sample(OT_OPT_LHS_Settings(n_samples=10))
    uncertain_coupling_graph.visualize(show=False, clean_up=False, file_path=file_name)
    check_dot_file(file_name)


def test_repeated_sampling(disciplines, uncertain_space, tmp_wd):
    """Check that the dispersion is estimated from all the samples."""
    file_name = "repeated_sampling.png"
    uncertain_coupling_graph = UncertainCouplingGraph(
        disciplines, uncertain_space, variable_names=["y_21"]
    )
    uncertain_coupling_graph.sample(OT_OPT_LHS_Settings(n_samples=10))
    uncertain_coupling_graph.sample(OT_OPT_LHS_Settings(n_samples=7, seed=3))
    uncertain_coupling_graph.visualize(show=False, clean_up=False, file_path=file_name)
    check_dot_file(file_name)


def test_self_coupled(tmp_wd):
    """Check the image computed with a self-coupled discipline."""
    file_name = "self_coupled.png"
    disciplines = [
        AnalyticDiscipline({"x": "(x+u)/2", "y": "u"}, name="D1"),
        AnalyticDiscipline({"z": "y"}, name="D2"),
    ]
    uncertain_space = DesignSpace()
    uncertain_space.add_variable("u", lower_bound=0.0, upper_bound=1)
    uncertain_coupling_graph = UncertainCouplingGraph(disciplines, uncertain_space)
    uncertain_coupling_graph.sample(OT_OPT_LHS_Settings(n_samples=10))
    uncertain_coupling_graph.visualize(
        show=False, clean_up=False, file_path=file_name, maximum_thickness=0.1
    )
    check_dot_file(file_name)


def test_several_couplings_per_edge(tmp_wd):
    """Check that the couplings of an edge are drawn in alphabetical order."""
    file_name = "several_couplings_per_edge.png"
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
    uncertain_coupling_graph.visualize(show=False, clean_up=False, file_path=file_name)
    check_dot_file(file_name)


@pytest.mark.parametrize("save", [False, True])
def test_visualize_save(uncertain_coupling_graph, save, tmp_wd):
    """Check visualize() with save option."""
    graph_view = uncertain_coupling_graph.visualize(show=False, save=save)
    assert isinstance(graph_view, GraphView)
    assert Path("graph_view.png").exists() is save


def test_no_deprecated_graph_view_import(monkeypatch):
    """Check that importing the module does not use `gemseo.post._graph_view`."""
    module_name = "gemseo_umdo.visualization.uncertain_coupling_graph"
    monkeypatch.delitem(sys.modules, "gemseo.post._graph_view", raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    package_name, attribute_name = module_name.rsplit(".", 1)
    monkeypatch.delattr(import_module(package_name), attribute_name, raising=False)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        import_module(module_name)

    assert not [r for r in records if "_graph_view" in str(r.message)]
