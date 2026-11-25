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

import pytest
from gemseo import compute_doe
from gemseo import sample_disciplines
from gemseo.dataset.io_dataset import IODataset
from gemseo.discipline import AnalyticDiscipline
from gemseo.machine_learning.regression.model import PCERegressor
from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
from gemseo.problem.uncertainty.ishigami import IshigamiSpace
from gemseo.problem.uncertainty.ishigami.statistics import sobol_1
from gemseo.problem.uncertainty.ishigami.statistics import sobol_2
from gemseo.problem.uncertainty.ishigami.statistics import sobol_3
from gemseo.problem.uncertainty.ishigami.statistics import sobol_12
from gemseo.problem.uncertainty.ishigami.statistics import sobol_13
from gemseo.problem.uncertainty.ishigami.statistics import sobol_23
from gemseo.problem.uncertainty.ishigami.statistics import total_sobol_1
from gemseo.problem.uncertainty.ishigami.statistics import total_sobol_2
from gemseo.problem.uncertainty.ishigami.statistics import total_sobol_3
from gemseo.uncertainty.sensitivity import SobolAnalysis
from gemseo.util.testing.helper import assert_exception
from graphviz import Digraph

from gemseo_umdo.visualization.sobol_graph import SobolGraph


@pytest.fixture(scope="module")
def first_order_indices() -> dict[str, float]:
    """The first-order Sobol' indices."""
    return {"X1": sobol_1, "X2": sobol_2, "X3": sobol_3}


@pytest.fixture(scope="module")
def total_order_indices() -> dict[str, float]:
    """The total-order Sobol' indices."""
    return {"X1": total_sobol_1, "X2": total_sobol_2, "X3": total_sobol_3}


@pytest.fixture(scope="module")
def second_order_indices() -> dict[tuple[str, str], float]:
    """The second-order Sobol' indices."""
    return {
        ("X1", "X2"): sobol_12,
        ("X1", "X3"): sobol_13,
        ("X2", "X3"): sobol_23,
    }


@pytest.fixture(scope="module")
def pce() -> PCERegressor:
    """A PCE."""
    samples = sample_disciplines(
        [IshigamiDiscipline(), AnalyticDiscipline({"y_copy": "y"})],
        IshigamiSpace(
            uniform_distribution_name=IshigamiSpace.UniformDistribution.OPENTURNS
        ),
        ["y", "y_copy"],
        algo_name="OT_OPT_LHS",
        n_samples=100,
    )
    pce_regressor = PCERegressor(samples)
    pce_regressor.learn()
    return pce_regressor


@pytest.fixture(scope="module")
def pce_with_vector_output() -> PCERegressor:
    """A PCE whose output y = x3 comes after an output a = [x1, x2] of size 2."""
    space = IshigamiSpace(
        uniform_distribution_name=IshigamiSpace.UniformDistribution.OPENTURNS
    )
    input_samples = compute_doe(space, algo_name="OT_OPT_LHS", n_samples=20)
    dataset = IODataset()
    dataset.add_input_group(input_samples, ["x1", "x2", "x3"])
    dataset.add_output_group(input_samples, ["a", "y"], {"a": 2, "y": 1})
    dataset.misc["input_space"] = space
    pce_regressor = PCERegressor(dataset)
    pce_regressor.learn()
    return pce_regressor


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"maximum_thickness": 30}, {"threshold": 0.5}],
    ids=["default", "maximum_thickness", "threshold"],
)
def test_draw(
    first_order_indices, total_order_indices, second_order_indices, kwargs, snapshot
):
    """Check the graph drawn by draw()."""
    digraph = SobolGraph(
        first_order_indices, total_order_indices, second_order_indices
    ).draw(**kwargs)
    assert isinstance(digraph, Digraph)
    assert digraph.source == snapshot


@pytest.fixture(scope="module")
def sobol_analysis() -> SobolAnalysis:
    """A Sobol' analysis of the Ishigami function."""
    analysis = SobolAnalysis()
    analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 100)
    analysis.compute_indices()
    return analysis


def test_from_analysis(sobol_analysis, snapshot):
    """Check the graph drawn from a Sobol' analysis."""
    assert SobolGraph.from_analysis(sobol_analysis, "y").draw().source == snapshot


_parametrize_errors = pytest.mark.parametrize(
    ("output_name", "output_component", "exception"),
    [
        ("z", 0, ValueError),
        ("y", 1, ValueError),
        ("y", -1, ValueError),
        ("y", 0.0, TypeError),
    ],
    ids=[
        "output_name",
        "output_component_too_large",
        "output_component_negative",
        "output_component_not_integer",
    ],
)


@_parametrize_errors
def test_from_analysis_error(
    sobol_analysis, output_name, output_component, exception, snapshot
):
    """Check the error raised by from_analysis with a wrong output name or component."""
    with assert_exception(exception, snapshot):
        SobolGraph.from_analysis(
            sobol_analysis, output_name, output_component=output_component
        )


@_parametrize_errors
def test_from_pce_error(pce, output_name, output_component, exception, snapshot):
    """Check the error raised by from_pce with a wrong output name or component."""
    with assert_exception(exception, snapshot):
        SobolGraph.from_pce(pce, output_name, output_component=output_component)


@pytest.mark.parametrize(
    ("pce_name", "output_name"),
    [("pce", "y"), ("pce", "y_copy"), ("pce_with_vector_output", "y")],
    ids=["y", "y_copy", "after_output_of_another_size"],
)
def test_from_pce(request, pce_name, output_name, snapshot):
    """Check the graph drawn from a PCE."""
    pce = request.getfixturevalue(pce_name)
    assert SobolGraph.from_pce(pce, output_name).draw().source == snapshot


@pytest.mark.parametrize("name", ["<A>", "ns:A"], ids=["html_like", "colon"])
def test_draw_special_name(name, snapshot):
    """Check that a name is neither an HTML-like label nor a node with a port."""
    digraph = SobolGraph(
        {name: 0.5, "B": 0.5}, {name: 0.6, "B": 0.6}, {(name, "B"): 0.3}
    ).draw()
    assert digraph.source == snapshot
