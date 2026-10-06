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
from typing import Final

from gemseo.core.function.array_function import ArrayFunction
from gemseo.formulation.core.base import BaseFormulation
from gemseo.uncertainty.statistic.core.base import BaseStatistics
from gemseo.util.constant import read_only_empty_dict
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.string import convert_strings_to_iterable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.base_factory import BaseFactory
    from gemseo.discipline import Discipline
    from gemseo.formulation.core.base_mdo import BaseMDOFormulation
    from gemseo.scenario import EvaluationScenario
    from gemseo.space import DesignSpace
    from gemseo.space import RandomSpace
    from gemseo.util.hashable_ndarray import HashableNdarray
    from gemseo.util.typing import RealArray
    from gemseo.util.typing import StrKeyMapping

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

    __DEFAULT_FACTOR: Final[float] = 2.0
    """The default factor related to the standard deviation."""

    __FACTOR_NAME: Final[str] = "factor"
    """The name of the argument for the factor related to the standard deviation."""

    __MARGIN_NAME: Final[str] = "Margin"
    """The name of the classes implementing the margin statistics."""

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

    _uncertain_space: RandomSpace
    """The uncertain space."""

    __available_statistics: list[str]
    """The names of the available statistics."""

    input_data_to_output_data: dict[HashableNdarray, dict[str, Any]]
    """The output samples or output statistics associated with the input data."""

    # TODO(bump-gemseo): pass the problem first, e.g. OptimizationProblem(design_space), then set its objective; the loose settings go into settings=<Formulation>_Settings(...)  # noqa: E501
    def __init__(
        self,
        disciplines: Sequence[Discipline | EvaluationScenario],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: RandomSpace,
        objective_statistic_name: str,
        settings: BaseUMDOFormulationSettings,
        minimize_objective: bool = True,
        objective_statistic_parameters: StrKeyMapping = read_only_empty_dict,
        mdo_formulation_settings: StrKeyMapping = read_only_empty_dict,
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

        if self._USE_AUXILIARY_MDO_FORMULATION:
            self._auxiliary_mdo_formulation = mdo_formulation.__class__(
                disciplines,
                objective_name,
                uncertain_space,
                **mdo_formulation_settings,
            )
        else:
            self._auxiliary_mdo_formulation = None

        objective_statistic_parameters = self.__update_statistic_parameters(
            objective_statistic_name,
            objective_statistic_parameters,
            not minimize_objective,
        )
        new_objective_name = self.__compute_name(
            objective_name,
            objective_statistic_name,
            **objective_statistic_parameters,
        )
        # TODO(bump-gemseo): pass the problem first, e.g. OptimizationProblem(design_space), then set its objective; the loose settings go into settings=<Formulation>_Settings(...)  # noqa: E501
        super().__init__(
            disciplines,
            new_objective_name,
            design_space,
            minimize_objective=minimize_objective,
            settings=settings,
        )
        self.name = f"{self.__class__.__name__}[{mdo_formulation.__class__.__name__}]"

        objective = self._statistic_function_class(
            self,
            mdo_formulation.problem.objective.name,
            ArrayFunction.FunctionType.OBJ,
            objective_statistic_name,
            **objective_statistic_parameters,
        )
        objective.name = new_objective_name
        self.problem.objective = objective
        self.problem.minimize_objective = minimize_objective

        # Initialize the cache mechanism.
        self.input_data_to_output_data = {}
        self.problem.add_listener(self._clear_input_data_to_output_data)

    @classmethod
    def __update_statistic_parameters(
        cls,
        statistic_name: str,
        statistic_parameters: StrKeyMapping,
        use_negative_factor: bool,
    ) -> StrKeyMapping:
        """Update the statistic parameters.

        Args:
            statistic_name: The name of the statistic.
            statistic_parameters: The statistic parameters to be updated.
            use_negative_factor: Whether to use a negative factor.

        Returns:
            The updated statistic parameters.
        """
        if statistic_name == cls.__MARGIN_NAME:
            statistic_parameters = statistic_parameters or {}
            factor = abs(
                statistic_parameters.get(cls.__FACTOR_NAME, cls.__DEFAULT_FACTOR)
            )
            if use_negative_factor:
                factor *= -1
            statistic_parameters[cls.__FACTOR_NAME] = factor

        return statistic_parameters

    def _build_objective(
        self,
        objective_name: str | Sequence[str],
        minimize_objective: bool,
        discipline: Discipline | None = None,
        top_level_disc: bool = True,
    ) -> None:
        return None

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
    def uncertain_space(self) -> RandomSpace:
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
        output_names = convert_strings_to_iterable(output_names)
        function_name = observable_name or "_".join(output_names)
        function_names = [
            function_.name
            for function_ in self._mdo_formulation.problem.functions
            if function_ is not None
        ]
        if function_name not in function_names:
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

        observable = self._statistic_function_class(
            self,
            function_name,
            ArrayFunction.FunctionType.NONE,
            statistic_name,
            **statistic_parameters,
        )
        observable.name = self.__compute_name(
            observable_name or output_names, statistic_name, **statistic_parameters
        )
        self.problem.add_observable(observable)
        self._post_add_observable()

    def add_constraint(
        self,
        output_name: str | Sequence[str],
        statistic_name: str,
        constraint_type: ArrayFunction.ConstraintType = ArrayFunction.ConstraintType.INEQ,
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
        function_name = "_".join(convert_strings_to_iterable(output_name))
        function_names = [
            function_.name
            for function_ in self._mdo_formulation.problem.functions
            if function_ is not None
        ]
        if function_name not in function_names:
            if self._auxiliary_mdo_formulation is not None:
                self._auxiliary_mdo_formulation.add_observable(output_name)

            self._mdo_formulation.add_observable(output_name)

        statistic_parameters = self.__update_statistic_parameters(
            statistic_name,
            statistic_parameters,
            positive,
        )
        constraint = self._statistic_function_class(
            self,
            function_name,
            ArrayFunction.FunctionType.NONE,
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
        self.problem.add_constraint(
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
            {n: v.size for n, v in self.input_space.variables.items()},
            list(self.input_space.variables),
        )
        for formulation in (self._mdo_formulation, self._auxiliary_mdo_formulation):
            if formulation is None:
                continue

            for discipline in formulation.get_top_level_disciplines(
                include_sub_formulations=True
            ):
                input_grammar = discipline.io.input_grammar
                to_value = input_grammar.data_converter.convert_array_to_value
                input_grammar.defaults.update({
                    name: to_value(name, value)
                    for name, value in design_values.items()
                    if name in input_grammar
                })

    def get_top_level_disciplines(self) -> list[Discipline]:  # noqa: D102
        return self._mdo_formulation.get_top_level_disciplines()
