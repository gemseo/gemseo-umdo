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
"""Scenarios to address multidisciplinary design problems under uncertainty."""
from __future__ import annotations

import logging
from typing import Any
from typing import Mapping
from typing import Sequence

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.formulation import MDOFormulation
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.scenario import Scenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.utils.python_compatibility import Final

from gemseo_umdo.formulations.factory import UMDOFormulationsFactory

LOGGER = logging.getLogger(__name__)


class _UScenario(Scenario):
    """Base scenario for multidisciplinary design problems under uncertainty."""

    __DV_TAG: Final[str] = "{}"
    __DV_PREFIX: Final[str] = "dv_"
    __MAXIMIZE_OBJECTIVE: Final[str] = "maximize_objective"

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        formulation: str,
        objective_name: str,
        design_space: DesignSpace,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        objective_statistic_parameters: Mapping[str, Any] | None = None,
        statistic_estimation: str = "Sampling",
        statistic_estimation_parameters: Mapping[str, Any] | None = None,
        uncertain_design_variables: Mapping[str, str] | None = None,
        name: str | None = None,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        **formulation_options: Any,
    ) -> None:
        """# noqa: D205 D212 D415
        Args:
            uncertain_space: The uncertain variables
                with their probability distributions.
            objective_statistic_name: The name of the statistic
                to be applied to the objective, e.g. "margin".
            objective_statistic_parameters: The parameters of the statistics
                to be applied to the objective,
                e.g. ``{"factor": 2.}`` when ``objective_statistic="margin"``.
            statistic_estimation: The name of the method to estimate the statistic.
            statistic_estimation_parameters: The options of ``statistic_estimation``.
            uncertain_design_variables: The expressions of the uncertainties
                applied to the design variables,
                e.g. ``{"x": "{} + u"}``
                where ``"x"`` is the name of the design variable
                actually used in the equations,
                ``"u"`` is the name of the uncertain variable
                defined in the ``uncertain_space``
                and ``"{}"`` is the optimization variable.
                Leave ``"{}"`` as is; it will be automatically replaced by ``"dv_x"``.
                If ``None``,
                do not consider other variable relations
                than those defined by ``disciplines``.
        """
        if statistic_estimation_parameters is None:
            statistic_estimation_parameters = {}

        maximize_objective = formulation_options.get(self.__MAXIMIZE_OBJECTIVE)
        if maximize_objective is not None:
            statistic_estimation_parameters[
                self.__MAXIMIZE_OBJECTIVE
            ] = maximize_objective

        formulations_factory = MDOFormulationsFactory()

        if uncertain_design_variables is not None:
            expressions = {}
            for dv_name, expression in uncertain_design_variables.items():
                new_dv_name = self.__DV_PREFIX + dv_name
                design_space.rename_variable(dv_name, new_dv_name)
                expressions[dv_name] = expression.replace(self.__DV_TAG, new_dv_name)

            disciplines.append(AnalyticDiscipline(expressions, "Design Uncertainties"))

        mdo_formulation = formulations_factory.create(
            formulation,
            disciplines,
            objective_name,
            uncertain_space,
            grammar_type=grammar_type,
            **formulation_options,
        )

        filtered_design_space = formulations_factory.create(
            formulation,
            disciplines,
            objective_name,
            design_space,
            **formulation_options,
        ).design_space

        super().__init__(
            disciplines,
            statistic_estimation,
            objective_name,
            filtered_design_space,
            name=name,
            mdo_formulation=mdo_formulation,
            objective_statistic_name=objective_statistic_name,
            objective_statistic_parameters=objective_statistic_parameters,
            uncertain_space=uncertain_space,
            **statistic_estimation_parameters,
        )

        self.formulation_name = self.formulation.name

    @property
    def _formulation_factory(self) -> UMDOFormulationsFactory:
        return UMDOFormulationsFactory()

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        statistic_name: str,
        constraint_type: str = MDOFunction.TYPE_INEQ,
        constraint_name: str | None = None,
        value: float | None = None,
        positive: bool = False,
        **statistic_parameters: Any,
    ) -> None:
        """# noqa: D205 D212 D415
        Args:
            statistic_name: The name of the statistic
                to be applied to the constraint, e.g. "margin".
            statistic_parameters: The parameters of the statistics
                to be applied to the constraint,
                ``{"factor": 2.}`` when ``objective_statistic="margin"``.
        """
        if constraint_type not in [MDOFunction.TYPE_EQ, MDOFunction.TYPE_INEQ]:
            raise ValueError(
                f"Constraint type must be either '{MDOFunction.TYPE_EQ}' "
                f"or '{MDOFunction.TYPE_INEQ}'; "
                f"got '{constraint_type}' instead."
            )
        self.formulation.add_constraint(
            output_name,
            statistic_name,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
            **statistic_parameters,
        )

    def add_observable(
        self,
        output_names: Sequence[str],
        statistic_name: str,
        observable_name: Sequence[str] | None = None,
        discipline: MDODiscipline | None = None,
        **statistic_parameters: Any,
    ) -> None:
        """# noqa: D205 D212 D415
        Args:
            statistic_name: The name of the statistic
                to be applied to the constraint, e.g. "margin".
            statistic_parameters: The parameters of the statistics
                to be applied to the constraint,
                ``{"factor": 2.}`` when ``objective_statistic="margin"``.
        """
        self.formulation.add_observable(
            output_names,
            statistic_name,
            observable_name=observable_name,
            discipline=discipline,
            **statistic_parameters,
        )

    def __repr__(self) -> str:
        msg = super().__repr__().split("\n")
        msg[2] = f"   Formulation: {self.formulation.name}"
        return "\n".join(msg)

    @property
    def uncertain_space(self) -> ParameterSpace:
        """The uncertain variable space."""
        return self.formulation.uncertain_space

    @property
    def mdo_formulation(self) -> MDOFormulation:
        """The MDO formulation."""
        return self.formulation.mdo_formulation

    @property
    def available_statistics(self) -> list[str]:
        """The names of the available statistics."""
        return self.formulation.available_statistics
