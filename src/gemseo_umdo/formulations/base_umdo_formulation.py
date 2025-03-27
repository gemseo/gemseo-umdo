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
"""Base class for U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.formulations.base_formulation import BaseFormulation
from gemseo.formulations.bilevel import BiLevel
from gemseo.uncertainty.statistics.base_statistics import BaseStatistics
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.file_path_manager import FilePathManager

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.hashable_ndarray import HashableNdarray
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.base_factory import BaseFactory
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping

    from gemseo_umdo.formulations._functions.base_statistic_function import (
        BaseStatisticFunction,
    )
    from gemseo_umdo.formulations.base_umdo_formulation_settings import (
        BaseUMDOFormulationSettings,
    )


class BaseUMDOFormulation(BaseFormulation):
    """Base class for U-MDO formulations.

    A U-MDO formulation rewrites a multidisciplinary optimization problem under
    uncertainty, a.k.a. U-MDO problem, as a standard optimization problem without
    uncertainty.
    """

    _USE_AUXILIARY_MDO_FORMULATION: ClassVar[bool] = False
    """Whether the U-MDO formulation uses an auxiliary MDO formulation.

    For this auxiliary formulation, the functions are evaluable over the uncertain space
    and differentiable with respect to the uncertain variables.
    """

    _mdo_formulation: BaseMDOFormulation
    """The MDO formulation.

    The functions are evaluable over the uncertain space and differentiable with respect
    to the design variables.
    """

    _auxiliary_mdo_formulation: BaseMDOFormulation | None
    """The auxiliary MDO formulation if :attr:`._USE_AUXILIARY_MDO_FORMULATION`.

    For this auxiliary formulation, the functions are evaluable over the uncertain space
    and differentiable with respect to the uncertain variables.
    """

    _statistic_factory: BaseFactory
    """A factory of statistics.

    Used only when `_STATISTIC_FACTORY` is `None`.

    To be used when a U-MDO formulation has several ways to estimate statistics
    and the choice is done at instantiation.
    For example, Sampling can estimate statistics in one go or iteratively.
    """

    _STATISTIC_FACTORY: ClassVar[BaseFactory]
    """A factory of statistics.

    If `None`, use `_statistic_factory`.

    To be used when a U-MDO formulation has only one way to estimate statistics.
    """

    _statistic_function_class: type[BaseStatisticFunction] | None
    """A subclass of `MDOFunction` to compute a statistic.

    Used only when `_STATISTIC_FUNCTION_CLASS` is `None`.

    To be used when a U-MDO formulation has several ways to estimate statistics
    and the choice is done at instantiation.
    For example, Sampling can estimate statistics in one go or iteratively.
    """

    _STATISTIC_FUNCTION_CLASS: ClassVar[type[BaseStatisticFunction] | None] = None
    """A subclass of `MDOFunction` to compute a statistic.

    If `None`, use `_statistic_function_class`.

    To be used when a U-MDO formulation has only one way to estimate statistics.
    """

    _uncertain_space: ParameterSpace
    """The uncertain space."""

    __available_statistics: list[str]
    """The names of the available statistics."""

    input_data_to_output_data: dict[HashableNdarray, dict[str, Any]]
    """The output samples or output statistics associated with the input data."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        settings_model: BaseUMDOFormulationSettings,
        objective_statistic_parameters: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        mdo_formulation_settings: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """
        Args:
            mdo_formulation: The MDO formulation
                generating functions evaluable over the uncertain space
                and differentiable with respect to the design variables.
            uncertain_space: The uncertain variables
                with their probability distributions.
            objective_statistic_name: The name of the statistic
                to be applied to the objective.
            objective_statistic_parameters: The values of the parameters
                of the statistic to be applied to the objective, if any.
            mdo_formulation_settings: The settings of the MDO formulation.
        """  # noqa: D205 D212 D415
        if self._STATISTIC_FUNCTION_CLASS is not None:
            self._statistic_function_class = self._STATISTIC_FUNCTION_CLASS
            self._statistic_factory = self._STATISTIC_FACTORY

        self.__available_statistics = self._statistic_factory.class_names
        self._mdo_formulation = mdo_formulation
        self._uncertain_space = uncertain_space

        # Create the auxiliary MDO formulation if required.
        self._auxiliary_mdo_formulation = None
        if self._USE_AUXILIARY_MDO_FORMULATION:
            self._auxiliary_mdo_formulation = mdo_formulation.__class__(
                disciplines,
                objective_name,
                uncertain_space,
                **mdo_formulation_settings,
            )

        # Create the objective name.
        objective_name = self.__compute_name(
            objective_name,
            objective_statistic_name,
            **objective_statistic_parameters,
        )
        super().__init__(
            disciplines, objective_name, design_space, settings_model=settings_model
        )
        self.name = f"{self.__class__.__name__}[{mdo_formulation.__class__.__name__}]"

        # Replace the objective function by a statistic function.
        sub_opt_problem = mdo_formulation.optimization_problem
        objective = self._statistic_function_class(
            self,
            sub_opt_problem.objective,
            MDOFunction.FunctionType.OBJ,
            objective_statistic_name,
            **objective_statistic_parameters,
        )
        objective.name = objective_name
        self.optimization_problem.objective = objective
        self.optimization_problem.minimize_objective = (
            mdo_formulation.optimization_problem.minimize_objective
        )

        # Initialize the cache mechanism.
        self.input_data_to_output_data = {}
        self.optimization_problem.add_listener(self._clear_input_data_to_output_data)

    def _clear_input_data_to_output_data(self, x_vect: RealArray) -> None:
        """Clear the attribute `input_data_to_output_data`.

        Args:
            x_vect: An input vector.
        """
        self.input_data_to_output_data.clear()

    @property
    def mdo_formulation(self) -> BaseMDOFormulation:
        """The MDO formulation.

        The functions are evaluable over the uncertain space and differentiable with
        respect to the design variables.
        """
        return self._mdo_formulation

    @property
    def auxiliary_mdo_formulation(self) -> BaseMDOFormulation:
        """The auxiliary MDO formulation.

        The functions are evaluable over the uncertain space and differentiable with
        respect to the uncertain variables.
        """
        return self._auxiliary_mdo_formulation

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
        observable_name: str = "",
        discipline: Discipline | None = None,
        **statistic_parameters: Any,
    ) -> None:
        """
        Args:
            statistic_name: The name of the statistic to be applied to the observable.
            statistic_parameters: The values of the parameters
                of the statistic to be applied to the observable, if any.
        """  # noqa: D205 D212 D415
        if self._auxiliary_mdo_formulation is not None:
            self._auxiliary_mdo_formulation.add_observable(
                output_names,
                observable_name=observable_name,
                discipline=discipline,
            )

        self._mdo_formulation.add_observable(
            output_names,
            observable_name=observable_name,
            discipline=discipline,
        )
        sub_opt_problem = self._mdo_formulation.optimization_problem
        observable = self._statistic_function_class(
            self,
            sub_opt_problem.observables[-1],
            MDOFunction.FunctionType.NONE,
            statistic_name,
            **statistic_parameters,
        )
        observable.name = self.__compute_name(
            observable_name or output_names, statistic_name, **statistic_parameters
        )
        self.optimization_problem.add_observable(observable)
        self._post_add_observable()

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        statistic_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.INEQ,
        constraint_name: str = "",
        value: float = 0.0,
        positive: bool = False,
        **statistic_parameters: Any,
    ) -> None:
        """
        Args:
            statistic_name: The name of the statistic to be applied to the constraint.
            statistic_parameters: The values of the parameters of the statistic
                to be applied to the constraint, if any.
        """  # noqa: D205 D212 D415
        if self._auxiliary_mdo_formulation is not None:
            self._auxiliary_mdo_formulation.add_observable(output_name)

        self._mdo_formulation.add_observable(output_name)
        constraint = self._statistic_function_class(
            self,
            self._mdo_formulation.optimization_problem.observables[-1],
            MDOFunction.FunctionType.NONE,
            statistic_name,
            **statistic_parameters,
        )
        name = self.__compute_name(output_name, statistic_name, **statistic_parameters)
        constraint.output_names = [name]
        if constraint_name:
            constraint.name = constraint_name
            constraint.has_default_name = False
        else:
            constraint.name = name
            constraint.has_default_name = True
        self.optimization_problem.add_constraint(
            constraint,
            value=value,
            positive=positive,
            constraint_type=constraint_type,
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

        return BaseStatistics.compute_expression(
            output_name,
            FilePathManager.to_snake_case(statistic_name),
            **statistic_parameters,
        )

    def update_top_level_disciplines(self, design_values: RealArray) -> None:
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
        for formulation in [self._mdo_formulation, self._auxiliary_mdo_formulation]:
            if formulation is None:
                continue

            all_top_level_disciplines = [formulation.get_top_level_disciplines()]
            # TODO: remove this block and find a more generic way of handling this case.
            if isinstance(formulation, BiLevel):
                all_top_level_disciplines.extend(
                    scenario_adapter.scenario.formulation.get_top_level_disciplines()
                    for scenario_adapter in formulation.scenario_adapters
                )

            for top_level_disciplines in all_top_level_disciplines:
                for discipline in top_level_disciplines:
                    input_grammar = discipline.io.input_grammar
                    to_value = input_grammar.data_converter.convert_array_to_value
                    input_grammar.defaults.update({
                        name: to_value(name, value)
                        for name, value in design_values.items()
                        if name in input_grammar
                    })

    def get_top_level_disciplines(self) -> list[Discipline]:  # noqa: D102
        return self._mdo_formulation.get_top_level_disciplines()
