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
"""Formulate a multidisciplinary design problem under uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.base_formulation import BaseFormulation
from gemseo.core.discipline import MDODiscipline
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.uncertainty.statistics.statistics import Statistics
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.base_factory import BaseFactory
    from gemseo.core.execution_sequence import ExecutionSequence
    from gemseo.core.formulation import MDOFormulation

    from gemseo_umdo.formulations.functions.statistic_function import StatisticFunction


class UMDOFormulation(BaseFormulation):
    """Base formulation of a multidisciplinary design problem under uncertainty."""

    _mdo_formulation: MDOFormulation
    """The MDO formulation used by the U-MDO formulation over the uncertain space."""

    _processed_functions: list[str]
    """The names of the functions whose statistics have been estimated."""

    _statistic_factory: BaseFactory
    """A factory of statistics."""

    _statistic_function_class: type[StatisticFunction]
    """A subclass of :class:`.MDOFunction` to compute a statistic."""

    _uncertain_space: ParameterSpace
    """The uncertain space."""

    __available_statistics: list[str]
    """The names of the available statistics."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: MDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        objective_statistic_parameters: Mapping[str, Any] | None = None,
        maximize_objective: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        **options: Any,
    ) -> None:
        """
        Args:
            mdo_formulation: The class name of the MDO formulation, e.g. "MDF".
            uncertain_space: The uncertain variables
                with their probability distributions.
            objective_statistic_name: The name of the statistic
                to be applied to the objective.
            objective_statistic_parameters: The values of the parameters
                of the statistic to be applied to the objective, if any.
        """  # noqa: D205 D212 D415
        pretty_str(design_space.variable_names)
        pretty_str(uncertain_space.variable_names)
        if objective_statistic_parameters is None:
            objective_statistic_parameters = {}

        objective_name = self.__compute_name(
            objective_name,
            objective_statistic_name,
            **objective_statistic_parameters,
        )
        self._uncertain_space = uncertain_space
        self._mdo_formulation = mdo_formulation
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            grammar_type=grammar_type,
            **options,
        )
        self.__available_statistics = self._statistic_factory.class_names
        sub_opt_problem = self._mdo_formulation.opt_problem
        objective = self._statistic_function_class(
            self,
            sub_opt_problem.objective,
            MDOFunction.FunctionType.OBJ,
            objective_statistic_name,
            sub_opt_problem,
            **objective_statistic_parameters,
        )
        objective.name = objective_name
        self.opt_problem.objective = objective
        self.opt_problem.minimize_objective = not maximize_objective
        self.name = f"{self.__class__.__name__}[{mdo_formulation.__class__.__name__}]"
        self._processed_functions = []

    @property
    def mdo_formulation(self) -> MDOFormulation:
        """The MDO formulation."""
        return self._mdo_formulation

    @property
    def uncertain_space(self) -> ParameterSpace:
        """The uncertain variable space."""
        return self._uncertain_space

    @property
    def available_statistics(self) -> list[str]:
        """The names of the statistics to quantify the output uncertainties."""
        return self.__available_statistics

    def add_observable(
        self,
        output_names: Sequence[str],
        statistic_name: str,
        observable_name: Sequence[str] | None = None,
        discipline: MDODiscipline | None = None,
        **statistic_parameters: Any,
    ) -> None:
        """
        Args:
            statistic_name: The name of the statistic to be applied to the observable.
            statistic_parameters: The values of the parameters
                of the statistic to be applied to the observable, if any.
        """  # noqa: D205 D212 D415
        self._mdo_formulation.add_observable(
            output_names,
            observable_name=observable_name,
            discipline=discipline,
        )
        sub_opt_problem = self._mdo_formulation.opt_problem
        observable = self._statistic_function_class(
            self,
            sub_opt_problem.observables[-1],
            MDOFunction.FunctionType.NONE,
            statistic_name,
            sub_opt_problem,
            **statistic_parameters,
        )
        observable.name = self.__compute_name(
            observable_name or output_names, statistic_name, **statistic_parameters
        )
        self.opt_problem.add_observable(observable)
        self._post_add_observable()

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        statistic_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.INEQ,
        constraint_name: str | None = None,
        value: float | None = None,
        positive: bool = False,
        **statistic_parameters: Any,
    ) -> None:
        """
        Args:
            statistic_name: The name of the statistic to be applied to the constraint.
            statistic_parameters: The values of the parameters of the statistic
                to be applied to the constraint, if any.
        """  # noqa: D205 D212 D415
        self._mdo_formulation.add_observable(output_name)
        sub_opt_problem = self._mdo_formulation.opt_problem
        constraint = self._statistic_function_class(
            self,
            sub_opt_problem.observables[-1],
            MDOFunction.FunctionType.NONE,
            statistic_name,
            sub_opt_problem,
            **statistic_parameters,
        )
        name = self.__compute_name(output_name, statistic_name, **statistic_parameters)
        constraint.output_names = [name]
        if constraint_name is None:
            constraint.name = name
            constraint.has_default_name = True
        else:
            constraint.name = constraint_name
            constraint.has_default_name = False
        self.opt_problem.add_constraint(
            constraint,
            value=value,
            positive=positive,
            cstr_type=constraint_type,
        )
        self._post_add_constraint()

    def _post_add_constraint(self) -> None:
        """Apply actions after adding a constraint."""

    def _post_add_observable(self) -> None:
        """Apply actions after adding an observable."""

    @staticmethod
    def __compute_name(
        output_name: str | Iterable[str],
        statistic_name: str,
        **statistic_parameters: Any,
    ) -> str:
        """Create the string representation of a statistic applied to output variables.

        Args:
            output_name: Either the names of the output variables
                for which to estimate the statistic or a unique name to define them.
            statistic_name: The name of the statistic to be applied to the variables.
            statistic_parameters: The values of the parameters of the statistic
                to be applied to the variable, if any.

        Returns:
            The string representations of the statistic applied to the output variables.
        """
        if not isinstance(output_name, str):
            output_name = "_".join(output_name)

        return Statistics.compute_expression(
            output_name,
            FilePathManager.to_snake_case(statistic_name),
            **statistic_parameters,
        )

    def update_top_level_disciplines(self, design_values: Mapping[str, Any]) -> None:
        """Update the default input values of the top-level disciplines.

        Args:
            design_values: The values of the design variables
                to update the default input values of the top-level disciplines.
        """
        design_values = split_array_to_dict_of_arrays(
            design_values,
            self.design_space.variable_sizes,
            self.design_space.variable_names,
        )
        for discipline in self._mdo_formulation.get_top_level_disc():
            discipline.default_inputs.update({
                k: v for k, v in design_values.items() if k in discipline.input_grammar
            })

    def get_top_level_disc(self) -> list[MDODiscipline]:  # noqa: D102
        return self._mdo_formulation.get_top_level_disc()

    def get_expected_workflow(  # noqa: D102
        self,
    ) -> list[ExecutionSequence, tuple[ExecutionSequence]]:
        return self._mdo_formulation.get_expected_workflow()

    def get_expected_dataflow(  # noqa: D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        return self._mdo_formulation.get_expected_dataflow()
