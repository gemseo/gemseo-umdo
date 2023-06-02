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

from pathlib import Path

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiDiscipline
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo_umdo.visualizations.uncertain_coupling_graph import (
    UncertainCouplingGraph,
)


@pytest.fixture(scope="module")
def uncertain_space() -> DesignSpace:
    """The uncertain space of the Sobieski's SSBJ problem."""
    design_space = SobieskiProblem().design_space
    design_variable_names = ["x_1", "x_2", "x_3", "x_shared"]
    design_space.filter(design_variable_names)
    optimum_design = split_array_to_dict_of_arrays(
        SobieskiProblem().optimum_design,
        design_space.variable_sizes,
        design_variable_names,
    )
    space = DesignSpace()
    for name, value in optimum_design.items():
        space.add_variable(
            name,
            size=value.size,
            l_b=value * 0.95,
            u_b=value * 1.05,
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
    graph.sample(10)
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


def test_output_names(disciplines, uncertain_space, tmp_wd):
    """Check the image computed with custom output names set at instantiation."""
    file_name = "output_names.png"
    uncertain_coupling_graph = UncertainCouplingGraph(
        disciplines, uncertain_space, variable_names=["y_21"]
    )
    uncertain_coupling_graph.sample(10)
    uncertain_coupling_graph.visualize(show=False, clean_up=False, file_path=file_name)
    check_dot_file(file_name)


def test_self_coupled(tmp_wd):
    """Check the image computed with a self-coupled discipline."""
    file_name = "self_coupled.png"
    disciplines = [
        AnalyticDiscipline({"x": "x+u", "y": "u"}, name="D1"),
        AnalyticDiscipline({"z": "y"}, name="D2"),
    ]
    uncertain_space = DesignSpace()
    uncertain_space.add_variable("u", l_b=0.0, u_b=1)
    uncertain_coupling_graph = UncertainCouplingGraph(disciplines, uncertain_space)
    uncertain_coupling_graph.sample(10)
    uncertain_coupling_graph.visualize(
        show=False, clean_up=False, file_path=file_name, maximum_thickness=0.1
    )
    check_dot_file(file_name)
