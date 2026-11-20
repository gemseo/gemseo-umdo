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
from gemseo import sample_disciplines
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


def test_from_analysis(snapshot):
    """Check the graph drawn from a Sobol' analysis."""
    analysis = SobolAnalysis()
    analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 100)
    analysis.compute_indices()
    assert SobolGraph.from_analysis(analysis, "y").draw().source == snapshot


def test_from_pce_error(pce, snapshot):
    """Check the error raised by from_pce when an output name does not exist."""
    with assert_exception(ValueError, snapshot):
        SobolGraph.from_pce(pce, "z")


@pytest.mark.parametrize("output_name", ["y", "y_copy"])
def test_from_pce(pce, output_name, snapshot):
    """Check the graph drawn from a PCE."""
    assert SobolGraph.from_pce(pce, output_name).draw().source == snapshot


def test_html_like_name(snapshot):
    """Check that a name between angle brackets is not an HTML-like label."""
    digraph = SobolGraph({"<A>": 0.5}, {"<A>": 0.6}, {}).draw()
    assert digraph.source == snapshot
