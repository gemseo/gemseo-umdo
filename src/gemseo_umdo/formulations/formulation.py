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

import logging
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Iterable
from typing import Mapping
from typing import Sequence

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.base_formulation import BaseFormulation
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequence
from gemseo.core.formulation import MDOFormulation
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.uncertainty.statistics.statistics import Statistics
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.file_path_manager import FilePathManager
from numpy import ndarray

from gemseo_umdo.estimators.estimator import BaseStatisticEstimatorFactory

LOGGER = logging.getLogger(__name__)


class UMDOFormulation(BaseFormulation):
    """Base formulation of a multidisciplinary design problem under uncertainty."""

    _processed_functions: list[str]
    """The names of the functions whose statistics have been estimated."""

    _STATISTIC_FACTORY: ClassVar = BaseStatisticEstimatorFactory()

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
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        **options: Any,
    ) -> None:
        """# noqa: D205 D212 D415
        Args:
            mdo_formulation: The class name of the MDO formulation, e.g. "MDF".
            uncertain_space: The uncertain variables
                with their probability distributions.
            objective_statistic_name: The name of the statistic
                to be applied to the objective.
            objective_statistic_parameters: The values of the parameters
                of the statistic to be applied to the objective, if any.
        """
        design_variables = ", ".join(design_space.variables_names)
        uncertain_variables = ", ".join(uncertain_space.variables_names)
        self.__signature = f"({design_variables}; {uncertain_variables})"
        if objective_statistic_parameters is None:
            objective_statistic_parameters = {}

        objective_expression, objective_name = self.__compute_name(
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
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
            **options,
        )
        self.__available_statistics = self._STATISTIC_FACTORY.classes
        self.opt_problem.objective = self._StatisticFunction(
            self,
            self._mdo_formulation.opt_problem.objective,
            MDOFunction.TYPE_OBJ,
            objective_statistic_name,
            **objective_statistic_parameters,
        )
        if self._maximize_objective:
            objective_name = f"-{objective_name}"
            self.opt_problem.minimize_objective = False

        self.opt_problem.objective.name = objective_name
        self.opt_problem.objective.special_repr = objective_expression
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
        """# noqa: D205 D212 D415
        Args:
            statistic_name: The name of the statistic to be applied to the observable.
            statistic_parameters: The values of the parameters
                of the statistic to be applied to the observable, if any.
        """
        self._mdo_formulation.add_observable(
            output_names,
            observable_name=observable_name,
            discipline=discipline,
        )
        observable = self._StatisticFunction(
            self,
            self._mdo_formulation.opt_problem.observables[-1],
            None,
            statistic_name,
            **statistic_parameters,
        )
        observable.special_repr, observable.name = self.__compute_name(
            observable_name or output_names, statistic_name, **statistic_parameters
        )
        self.opt_problem.add_observable(observable)
        self._post_add_observable()

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        statistic_name: str,
        constraint_type: str = MDOFunction.TYPE_INEQ,
        constraint_name: str | None = None,
        value: float | None = None,
        positive: bool = False,
        **statistic_parameters,
    ) -> None:
        """# noqa: D205 D212 D415
        Args:
            statistic_name: The name of the statistic to be applied to the constraint.
            statistic_parameters: The values of the parameters of the statistic
                to be applied to the constraint, if any.
        """
        self._mdo_formulation.add_constraint(
            output_name,
            constraint_name=constraint_name,
        )
        constraint = self._StatisticFunction(
            self,
            self._mdo_formulation.opt_problem.constraints[-1],
            None,
            statistic_name,
            **statistic_parameters,
        )
        constraint.name, constraint.special_repr = self.__compute_name(
            constraint_name or output_name, statistic_name, **statistic_parameters
        )
        self.opt_problem.add_constraint(
            constraint,
            value=value,
            positive=positive,
            cstr_type=constraint_type,
        )
        self._post_add_constraint()

    def _post_add_constraint(self):
        """Apply actions after adding a constraint."""

    def _post_add_observable(self):
        """Apply actions after adding an observable."""

    def __compute_name(
        self,
        output_name: str | Iterable[str],
        statistic_name: str,
        **statistic_parameters: Any,
    ) -> tuple[str, str]:
        """Create the string representation of a statistic applied to output variables.

        Args:
            output_name: Either the names of the output variables
                for which to estimate the statistic or a unique name to define them.
            statistic: The name of the statistic to be applied to the variables.
            statistic_parameters: The values of the parameters of the statistic
                to be applied to the variable, if any.

        Returns:
            The string representations of the statistic applied to the variables,
            with and without the signature.
        """
        if not isinstance(output_name, str):
            output_name = "_".join(output_name)

        statistic_name = FilePathManager.to_snake_case(statistic_name)
        name_with_signature = Statistics.compute_expression(
            f"{output_name}{self.__signature}", statistic_name, **statistic_parameters
        )
        name = Statistics.compute_expression(
            output_name, statistic_name, **statistic_parameters
        )
        return name_with_signature, name

    def update_top_level_disciplines(self, design_values: ndarray) -> None:
        """Update the default input values of the top-level disciplines.

        Args:
            design_values: The values of the design variables
                to update the default input values of the top-level disciplines.
        """
        design_values = split_array_to_dict_of_arrays(
            design_values,
            self.design_space.variables_sizes,
            self.design_space.variables_names,
        )
        for discipline in self._mdo_formulation.get_top_level_disc():
            discipline.default_inputs.update(design_values)

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

    class _StatisticFunction(MDOFunction):
        """Compute a statistic of a function."""

        _estimate_statistic: Callable
        """The function to estimate the statistic."""

        _function_name: str
        """The name of the function."""

        _formulation: UMDOFormulation
        """The U-MDO formulation."""

        _statistic_parameters: dict[str, Any]
        """The parameters of the statistic."""

        def __init__(
            self,
            formulation: UMDOFormulation,
            func: MDOFunction,
            function_type: str,
            name: str,
            **parameters: Any,
        ) -> None:
            """# noqa: D205 D212 D415
            Args:
                formulation: The U-MDO formulation.
                func: The function for which to calculate the statistic.
                function_type: The type of function.
                name: The name of the statistic.
                **parameters: The parameters of the statistic.
            """
            self._estimate_statistic = formulation._STATISTIC_FACTORY.create(
                name, formulation=formulation
            )
            self._function_name = func.name
            self._formulation = formulation
            self._statistic_parameters = parameters
            super().__init__(self._func, name=func.name, f_type=function_type)

        @abstractmethod
        def _func(self, input_data: ndarray) -> ndarray:
            """A function computing output data from input data.

            Args:
                input_data: The input data of the function.

            Returns:
                The output data of the function.
            """
            ...
