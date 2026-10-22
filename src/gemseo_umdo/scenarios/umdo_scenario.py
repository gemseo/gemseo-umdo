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
from typing import Any

from gemseo.core.function.array_function import ArrayFunction
from gemseo.scenario import MDOScenario

from gemseo_umdo.scenarios.base_u_scenario import BaseUScenario

if TYPE_CHECKING:
    from collections.abc import Sequence


class UMDOScenario(BaseUScenario, MDOScenario):
    """A scenario for multidisciplinary design optimization under uncertainty.

    It solves the U-MDO problem with an optimizer
    or samples its objective, constraints and observables with a DOE algorithm.
    """

    def add_objective(
        self,
        output_names: str | Sequence[str],
        statistic_name: str,
        minimize: bool = True,
        objective_name: str = "",
        **statistic_parameters: Any,
    ) -> None:
        """Set the objective.

        Args:
            output_names: The names of the outputs used as objective.
            statistic_name: The name of the statistic of `output_names`, e.g. "Margin".
            minimize: Whether to minimize the statistic of the objective.
            objective_name: The name of the objective to be stored.
                If empty,
                the name is generated from `output_names` and `statistic_name`.
            **statistic_parameters: The parameters of the statistic,
                e.g. `factor=2.` when `statistic_name="Margin"`.
        """
        problem = self.formulation.problem
        problem.objective = self.formulation.create_objective(
            output_names,
            statistic_name,
            objective_name=objective_name,
            minimize=minimize,
            **statistic_parameters,
        )
        problem.minimize_objective = minimize

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        statistic_name: str,
        constraint_type: ArrayFunction.ConstraintType = (
            ArrayFunction.ConstraintType.INEQ
        ),
        constraint_name: str = "",
        value: float = 0,
        positive: bool = False,
        **statistic_parameters: Any,
    ) -> None:
        """
        Args:
            statistic_name: The name of the statistic of `output_name`, e.g. "Margin".
            **statistic_parameters: The parameters of the statistic,
                e.g. `factor=2.` when `statistic_name="Margin"`.
        """  # noqa: D205 D212 D415
        self.formulation.add_constraint(
            output_name,
            statistic_name,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
            **statistic_parameters,
        )
