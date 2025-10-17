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

from typing import TYPE_CHECKING

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.openturns.settings.ot_monte_carlo import OT_MONTE_CARLO_Settings
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline.discipline import Discipline
from gemseo.mlearning.regression.algos.pce_settings import PCERegressor_Settings
from numpy import diag
from numpy import hstack
from numpy import linspace
from numpy import newaxis
from numpy import ones
from numpy import vstack
from numpy.testing import assert_allclose

from gemseo_umdo.formulations.pce_settings import PCE_Settings
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


class A(Discipline):
    def __init__(self, n_x: int, n_u: int):
        super().__init__()
        self.m = diag(linspace(1, n_x + n_u, n_x + n_u))
        self.m_x = self.m[:, :n_x]
        self.m_u = self.m[n_x : n_x + n_u, n_x : n_x + n_u]
        self.input_grammar.update_from_names(["x", "u"])
        self.output_grammar.update_from_names(["y"])
        self.n_x = n_x
        self.n_u = n_u

    def _run(self, input_data: StrKeyMapping) -> None:
        x = self.io.data["x"]
        u = self.io.data["u"]
        self.io.update_output_data({"y": self.m @ hstack((x * u.sum(), u))})

    def _compute_jacobian(self, inputs=None, outputs=None) -> None:
        x = self.io.data["x"]
        u = self.io.data["u"]
        dydx = self.m_x * u.sum()
        dydu = vstack((
            (x * linspace(1, self.n_x, self.n_x)).reshape(-1, 1) * ones((1, self.n_u)),
            self.m_u,
        ))
        self.jac = {"y": {"x": dydx, "u": dydu}}


@pytest.mark.parametrize(
    ("statistic", "symbol"),
    [
        ("Mean", "E"),
        ("Variance", "V"),
        ("StandardDeviation", "StD"),
        ("Margin", "Margin"),
    ],
)
@pytest.mark.parametrize("n_x", [1, 2])
@pytest.mark.parametrize("n_u", [1, 2])
@pytest.mark.parametrize(
    "settings",
    [
        Sampling_Settings(
            doe_algo_settings=OT_MONTE_CARLO_Settings(n_samples=5),
        ),
        Sampling_Settings(
            doe_algo_settings=OT_MONTE_CARLO_Settings(n_samples=5),
            estimate_statistics_iteratively=False,
        ),
        PCE_Settings(regressor_settings=PCERegressor_Settings(degree=1)),
    ],
)
def test_derivatives(statistic, symbol, n_x, n_u, settings):
    """Check the analytical derivatives with different estimation techniques."""
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("u", "OTNormalDistribution", size=n_u)

    discipline = A(n_x, n_u)

    design_space = DesignSpace()
    design_space.add_variable("x", size=n_x, lower_bound=0.0, upper_bound=2.0)

    scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        statistic,
        formulation_name="DisciplinaryOpt",
        statistic_estimation_settings=settings,
    )
    scenario.execute(
        algo_name="CustomDOE", samples=linspace(1, n_x, n_x)[newaxis, :], eval_jac=True
    )
    last_item = scenario.formulation.optimization_problem.database.last_item

    scenario = UDOEScenario(
        [discipline],
        "y",
        design_space,
        uncertain_space,
        statistic,
        formulation_name="DisciplinaryOpt",
        statistic_estimation_settings=settings,
    )
    scenario.set_differentiation_method("finite_differences")
    scenario.execute(
        algo_name="CustomDOE", samples=linspace(1, n_x, n_x)[newaxis, :], eval_jac=True
    )
    new_last_item = scenario.formulation.optimization_problem.database.last_item
    name = "@Margin[y; 2.0]" if statistic == "Margin" else f"@{symbol}[y]"
    assert_allclose(
        new_last_item[name],
        last_item[name],
        **{"atol": 1} if isinstance(settings, PCE_Settings) else {"rtol": 1e-5},
    )
