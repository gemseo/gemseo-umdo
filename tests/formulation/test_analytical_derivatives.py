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
from typing import Final

import pytest
from gemseo.discipline import Discipline
from gemseo.doe import CustomDOE_Settings
from gemseo.doe import OT_MONTE_CARLO_Settings
from gemseo.machine_learning import PCERegressor_Settings
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from numpy import diag
from numpy import hstack
from numpy import linspace
from numpy import newaxis
from numpy import ones
from numpy import vstack
from numpy.testing import assert_allclose

from gemseo_umdo.formulation.pce_settings import PCE_Settings
from gemseo_umdo.formulation.sampling_settings import Sampling_Settings
from gemseo_umdo.scenario.mdo import UMDOScenario

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping

    from gemseo_umdo.formulation.base_umdo_formulation_settings import (
        BaseUMDOFormulationSettings,
    )


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
        x = self.io.input_data["x"]
        u = self.io.input_data["u"]
        self.io.update_output_data({"y": self.m @ hstack((x * u.sum(), u))})

    def _compute_jacobian(self, inputs=None, outputs=None) -> None:
        x = self.io.input_data["x"]
        u = self.io.input_data["u"]
        dydx = self.m_x * u.sum()
        dydu = vstack((
            (x * linspace(1, self.n_x, self.n_x)).reshape(-1, 1) * ones((1, self.n_u)),
            self.m_u,
        ))
        self.jac = {"y": {"x": dydx, "u": dydu}}


_statistics: Final[tuple[tuple[str, str], ...]] = (
    ("Mean", "E[y]"),
    ("Variance", "V[y]"),
    ("StandardDeviation", "StD[y]"),
    ("Margin", "Margin[y; 2.0]"),
    ("Quantile", "Q[y; 0.95]"),
)
"""The statistics to check, with the names of the functions computing them."""

_settings: Final[tuple[tuple[str, BaseUMDOFormulationSettings], ...]] = (
    (
        "iterative_sampling",
        Sampling_Settings(
            doe_algo_settings=OT_MONTE_CARLO_Settings(n_samples=5),
        ),
    ),
    (
        "sampling",
        Sampling_Settings(
            doe_algo_settings=OT_MONTE_CARLO_Settings(n_samples=5),
            estimate_statistics_iteratively=False,
        ),
    ),
    ("pce", PCE_Settings(regressor_settings=PCERegressor_Settings(degree=1))),
)
"""The settings of the U-MDO formulations to check, with their test IDs."""


@pytest.mark.parametrize(
    ("statistic", "name", "settings"),
    [
        pytest.param(statistic, name, settings, id=f"{statistic}-{settings_id}")
        for statistic, name in _statistics
        for settings_id, settings in _settings
        # The quantile is available neither for PCE
        # nor for the iterative sampling.
        if not (statistic == "Quantile" and settings_id != "sampling")
    ],
)
@pytest.mark.parametrize("n_x", [1, 2])
@pytest.mark.parametrize("n_u", [1, 2])
def test_derivatives(statistic, name, n_x, n_u, settings):
    """Check the analytical derivatives with different estimation techniques."""
    uncertain_space = RandomSpace()
    uncertain_space.add_variable("u", *[OTNormalDistribution_Settings()] * n_u)

    discipline = A(n_x, n_u)

    design_space = DesignSpace()
    design_space.add_variable("x", size=n_x, lower_bound=0.0, upper_bound=2.0)

    scenario = UMDOScenario(
        [discipline],
        design_space,
        uncertain_space,
        settings,
    )
    scenario.add_objective("y", statistic)
    scenario.execute(
        CustomDOE_Settings(samples=linspace(1, n_x, n_x)[newaxis, :], eval_jac=True)
    )
    last_item = scenario.formulation.problem.database.last_item

    scenario = UMDOScenario(
        [discipline],
        design_space,
        uncertain_space,
        settings,
    )
    scenario.add_objective("y", statistic)
    scenario.set_differentiation_method("finite_differences")
    scenario.execute(
        CustomDOE_Settings(samples=linspace(1, n_x, n_x)[newaxis, :], eval_jac=True)
    )
    new_last_item = scenario.formulation.problem.database.last_item
    assert_allclose(
        new_last_item[f"@{name}"],
        last_item[f"@{name}"],
        **{"atol": 1} if isinstance(settings, PCE_Settings) else {"rtol": 1e-5},
    )
