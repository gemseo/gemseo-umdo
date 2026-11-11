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
"""Scenario for multidisciplinary design optimization problems under uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.function.array_function import ArrayFunction
from gemseo.scenario import MDOScenario

from gemseo_umdo.scenario.base import BaseUScenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo_umdo.statistic.core.base_settings import BaseStatisticSettings


class UMDOScenario(BaseUScenario, MDOScenario):
    """A scenario for multidisciplinary design optimization under uncertainty.

    It solves the U-MDO problem with an optimizer
    or samples its objective, constraints and observables with a DOE algorithm.
    """

    def add_objective(
        self,
        output_names: str | Sequence[str],
        statistic_settings: str | BaseStatisticSettings,
        minimize: bool = True,
        objective_name: str = "",
    ) -> None:
        """Set the objective.

        Args:
            output_names: The names of the outputs used as objective.
            statistic_settings: Either the settings of the statistic of
                `output_names`, e.g. `Margin_Settings(factor=3.0)`,
                or the case-insensitive name of the statistic
                to use its default settings,
                e.g. `"Mean"`.
            minimize: Whether to minimize the statistic of the objective.
            objective_name: The name of the objective to be stored.
                If empty,
                the name is generated from `output_names` and `statistic_settings`.
        """
        problem = self.formulation.problem
        problem.objective = self.formulation.create_objective(
            output_names,
            statistic_settings,
            objective_name=objective_name,
            minimize=minimize,
        )
        problem.minimize_objective = minimize

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        statistic_settings: str | BaseStatisticSettings,
        constraint_type: ArrayFunction.ConstraintType = (
            ArrayFunction.ConstraintType.INEQ
        ),
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
    ) -> None:
        """
        Args:
            statistic_settings: Either the settings of the statistic of
                `output_name`, e.g. `Margin_Settings(factor=3.0)`,
                or the case-insensitive name of the statistic
                to use its default settings,
                e.g. `"Mean"`.
        """  # noqa: D205 D212 D415
        self.formulation.add_constraint(
            output_name,
            statistic_settings,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
        )
