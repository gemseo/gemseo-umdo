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
from pathlib import Path

import pytest
from gemseo import sample_disciplines
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.pce import PCERegressor
from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_1
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_2
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_3
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_12
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_13
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_23
from gemseo.problems.uncertainty.ishigami.statistics import TOTAL_SOBOL_1
from gemseo.problems.uncertainty.ishigami.statistics import TOTAL_SOBOL_2
from gemseo.problems.uncertainty.ishigami.statistics import TOTAL_SOBOL_3
from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis

from gemseo_umdo.visualizations.sobol_graph import SobolGraph


@pytest.fixture(scope="module")
def first_order_indices() -> dict[str, float]:
    """The first-order Sobol' indices."""
    return {"X1": SOBOL_1, "X2": SOBOL_2, "X3": SOBOL_3}


@pytest.fixture(scope="module")
def total_order_indices() -> dict[str, float]:
    """The total-order Sobol' indices."""
    return {"X1": TOTAL_SOBOL_1, "X2": TOTAL_SOBOL_2, "X3": TOTAL_SOBOL_3}


@pytest.fixture(scope="module")
def second_order_indices() -> dict[tuple[str, str], float]:
    """The second-order Sobol' indices."""
    return {
        ("X1", "X2"): SOBOL_12,
        ("X1", "X3"): SOBOL_13,
        ("X2", "X3"): SOBOL_23,
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
        == (Path(__file__).parent / "sobol_graph" / file_path).read_text().strip()
    )


def test_default(
    first_order_indices, total_order_indices, second_order_indices, tmp_wd
):
    """Check the image computed with the default configuration."""
    file_name = "default.png"
    SobolGraph(
        first_order_indices,
        total_order_indices,
        second_order_indices,
    ).visualize(show=False, clean_up=False, file_path=file_name)
    check_dot_file(file_name)


def test_maximum_thickness(
    first_order_indices, total_order_indices, second_order_indices, tmp_wd
):
    """Check the image computed with a custom maximum thickness."""
    file_name = "maximum_thickness.png"
    SobolGraph(
        first_order_indices,
        total_order_indices,
        second_order_indices,
        maximum_thickness=30,
    ).visualize(show=False, clean_up=False, file_path=file_name)
    check_dot_file(file_name)


def test_threshold(
    first_order_indices, total_order_indices, second_order_indices, tmp_wd
):
    """Check the image computed with a custom threshold."""
    file_name = "threshold.png"
    SobolGraph(
        first_order_indices,
        total_order_indices,
        second_order_indices,
        threshold=0.5,
    ).visualize(show=False, clean_up=False, file_path=file_name)
    check_dot_file(file_name)


def test_from_analysis(tmp_wd):
    """Check the image computed from a Sobol' analysis."""
    analysis = SobolAnalysis()
    analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 100)
    analysis.compute_indices()
    file_name = "from_analysis.png"
    SobolGraph.from_analysis(analysis, "y").visualize(
        show=False, clean_up=False, file_path=file_name
    )
    check_dot_file(file_name)


def test_from_pce_error(pce):
    """Check the error raised by from_pce when an output name does not exist."""
    with pytest.raises(
        ValueError, match=re.escape("The name 'z' is not an output name.")
    ):
        SobolGraph.from_pce(pce, "z")


@pytest.mark.parametrize("output_name", ["y", "y_copy"])
def test_from_pce(tmp_wd, pce, output_name):
    """Check the image computed from a PCE."""
    file_name = "from_pce.png"
    SobolGraph.from_pce(pce, output_name).visualize(
        show=False, clean_up=False, file_path=file_name
    )
    check_dot_file(file_name)
