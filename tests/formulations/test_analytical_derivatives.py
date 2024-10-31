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
"""Test functions for the use of analytical derivatives."""

from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline.discipline import Discipline
from numpy import array
from numpy import diag
from numpy import hstack
from numpy import linspace
from numpy import newaxis
from numpy.testing import assert_almost_equal

from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario


class A(Discipline):
    def __init__(self, n_x: int, n_u: int):
        super().__init__()
        self.m = diag(linspace(1, n_x + n_u, n_x + n_u))
        self.m_x = self.m[:, :n_x]
        self.m_u = self.m[:, n_x : n_x + n_u]
        self.input_grammar.update_from_names(["x", "u"])
        self.output_grammar.update_from_names(["y"])
        self.n_x = n_x
        self.n_u = n_u

    def _run(self) -> None:
        x = self.io.data["x"]
        u = self.io.data["u"]
        self.io.update_output_data({"y": self.m @ hstack((x * u.sum(), u))})

    def _compute_jacobian(self, inputs=None, outputs=None) -> None:
        u = self.io.data["u"]
        self.jac = {"y": {"x": self.m_x * u.sum(), "u": self.m_u}}


@pytest.mark.parametrize("estimate_statistics_iteratively", [False, True])
@pytest.mark.parametrize(
    (
        "n_x",
        "n_u",
        "statistic",
        "symbol",
        "statistic_estimation",
        "statistic_estimation_jacobian",
    ),
    [
        (1, 1, "Mean", "E", array([1.5, 3.0]), array([[1.5], [0.0]])),
        (1, 2, "Mean", "E", array([4.0, 3.0, 7.5]), array([[4.0], [0.0], [0.0]])),
        (
            2,
            1,
            "Mean",
            "E",
            array([1.5, 6.0, 4.5]),
            array([[1.5, 0.0], [0.0, 3.0], [0.0, 0.0]]),
        ),
        (
            2,
            2,
            "Mean",
            "E",
            array([4.0, 16.0, 4.5, 10.0]),
            array([[4.0, 0.0], [0.0, 8.0], [0.0, 0.0], [0.0, 0.0]]),
        ),
        (1, 1, "Variance", "V", array([0.5, 2.0]), array([[1.0], [0.0]])),
        (1, 2, "Variance", "V", array([2.0, 2.0, 4.5]), array([[4.0], [0.0], [0.0]])),
        (
            2,
            1,
            "Variance",
            "V",
            array([0.5, 8.0, 4.5]),
            array([[1.0, 0.0], [0.0, 8.0], [0.0, 0.0]]),
        ),
        (
            2,
            2,
            "Variance",
            "V",
            array([2.0, 32.0, 4.5, 8.0]),
            array([[4.0, 0.0], [0.0, 32.0], [0.0, 0.0], [0.0, 0.0]]),
        ),
        (
            1,
            1,
            "StandardDeviation",
            "StD",
            array([0.5, 2.0]) ** 0.5,
            array([[1.0], [0.0]]) / array([[0.5], [2.0]]) ** 0.5 / 2,
        ),
        (
            1,
            2,
            "StandardDeviation",
            "StD",
            array([2.0, 2.0, 4.5]) ** 0.5,
            array([[4.0], [0.0], [0.0]]) / array([[2.0], [2.0], [4.5]]) ** 0.5 / 2,
        ),
        (
            2,
            1,
            "StandardDeviation",
            "StD",
            array([0.5, 8.0, 4.5]) ** 0.5,
            array([[1.0, 0.0], [0.0, 8.0], [0.0, 0.0]])
            / array([[0.5], [8.0], [4.5]]) ** 0.5
            / 2,
        ),
        (
            2,
            2,
            "StandardDeviation",
            "StD",
            array([2.0, 32.0, 4.5, 8.0]) ** 0.5,
            array([[4.0, 0.0], [0.0, 32.0], [0.0, 0.0], [0.0, 0.0]])
            / array([[2.0], [32.0], [4.5], [8.0]]) ** 0.5
            / 2,
        ),
        (
            1,
            1,
            "Margin",
            "Margin",
            array([1.5, 3.0]) + 2 * array([0.5, 2.0]) ** 0.5,
            array([[1.5], [0.0]])
            + 2 * array([[1.0], [0.0]]) / array([[0.5], [2.0]]) ** 0.5 / 2,
        ),
        (
            1,
            2,
            "Margin",
            "Margin",
            array([4.0, 3.0, 7.5]) + 2 * array([2.0, 2.0, 4.5]) ** 0.5,
            array([[4.0], [0.0], [0.0]])
            + 2
            * array([[4.0], [0.0], [0.0]])
            / array([[2.0], [2.0], [4.5]]) ** 0.5
            / 2,
        ),
        (
            2,
            1,
            "Margin",
            "Margin",
            array([1.5, 6.0, 4.5]) + 2 * array([0.5, 8.0, 4.5]) ** 0.5,
            array([[1.5, 0.0], [0.0, 3.0], [0.0, 0.0]])
            + 2
            * array([[1.0, 0.0], [0.0, 8.0], [0.0, 0.0]])
            / array([[0.5], [8.0], [4.5]]) ** 0.5
            / 2,
        ),
        (
            2,
            2,
            "Margin",
            "Margin",
            array([4.0, 16.0, 4.5, 10.0]) + 2 * array([2.0, 32.0, 4.5, 8.0]) ** 0.5,
            array([[4.0, 0.0], [0.0, 8.0], [0.0, 0.0], [0.0, 0.0]])
            + 2
            * array([[4.0, 0.0], [0.0, 32.0], [0.0, 0.0], [0.0, 0.0]])
            / array([[2.0], [32.0], [4.5], [8.0]]) ** 0.5
            / 2,
        ),
    ],
)
def test_sampling(
    estimate_statistics_iteratively,
    statistic,
    symbol,
    n_x,
    n_u,
    statistic_estimation,
    statistic_estimation_jacobian,
):
    """Check the Jacobian of the MC estimators."""
    discipline = A(n_x, n_u)

    design_space = DesignSpace()
    design_space.add_variable("x", size=n_x, lower_bound=0.0, upper_bound=2.0)

    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("u", "OTNormalDistribution", size=n_u)

    scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        statistic,
        formulation_name="DisciplinaryOpt",
        statistic_estimation_parameters={
            "algo": "CustomDOE",
            "algo_options": {
                "samples": array([
                    linspace(1, n_u, n_u).tolist(),
                    linspace(2, n_u + 1, n_u).tolist(),
                ])
            },
            "estimate_statistics_iteratively": estimate_statistics_iteratively,
        },
    )
    scenario.execute(
        algo_name="CustomDOE", samples=linspace(1, n_x, n_x)[newaxis, :], eval_jac=True
    )
    last_item = scenario.formulation.optimization_problem.database.last_item
    assert_almost_equal(last_item[f"{symbol}[y]"], statistic_estimation)
    assert_almost_equal(last_item[f"@{symbol}[y]"], statistic_estimation_jacobian)
