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
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.formulation import MDF_Settings
from gemseo.formulation.core.base import BaseFormulation
from gemseo.formulation.factory import mdo_formulation_factory
from gemseo.optimization.core.constraints import Constraints
from gemseo.space import DesignSpace
from gemseo.uncertainty.statistic.core.base import BaseStatistics
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.hashable_ndarray import HashableNdarray
from gemseo.util.string import convert_strings_to_iterable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.base_factory import BaseFactory
    from gemseo.discipline import Discipline
    from gemseo.formulation.core.base_mdo import BaseMDOFormulation
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.optimization import OptimizationProblem
    from gemseo.scenario import EvaluationScenario
    from gemseo.space import RandomSpace
    from gemseo.util.typing import RealArray
    from gemseo.util.typing import StrKeyMapping

    from gemseo_umdo.formulations._functions.base_statistic_function import (
        BaseStatisticFunction,
    )
    from gemseo_umdo.formulations.base_umdo_formulation_settings import (
        BaseUMDOFormulationSettings,
    )


_default_factor: Final[float] = 2.0
"""The default factor related to the standard deviation."""

_factor_name: Final[str] = "factor"
"""The name of the argument for the factor related to the standard deviation."""

_margin_name: Final[str] = "Margin"
"""The name of the classes implementing the margin statistics."""


class BaseUMDOFormulation(BaseFormulation["BaseUMDOFormulationSettings", DesignSpace]):
    """Base class for U-MDO formulations.

    A U-MDO formulation rewrites a multidisciplinary optimization problem under
    uncertainty, a.k.a. U-MDO problem, as a standard optimization problem without
    uncertainty.
    """

    _use_auxiliary_mdo_formulation: ClassVar[bool] = False
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
    """The auxiliary MDO formulation if `_use_auxiliary_mdo_formulation`.

    For this auxiliary formulation, the functions are evaluable over the uncertain space
    and differentiable with respect to the uncertain variables.
    """

    _statistic_factory: BaseFactory
    """A factory of statistics.

    A U-MDO formulation with only one way to estimate statistics sets it at class level.
    A U-MDO formulation with several ways sets it at instantiation,
    before calling the `__init__` of this class.
    For example, Sampling can estimate statistics in one go or iteratively.
    """

    _statistic_function_class: type[BaseStatisticFunction]
    """A subclass of `ArrayFunction` to compute a statistic.

    Set together with `_statistic_factory`.
    """

    _uncertain_space: RandomSpace
    """The uncertain space."""

    __available_statistics: list[str]
    """The names of the available statistics."""

    input_data_to_output_data: dict[HashableNdarray, dict[bool, dict[str, Any]]]
    """The output samples or output statistics associated with the input data.

    For each input point,
    the data to estimate the statistics are stored with the key `False`
    and the data to estimate their Jacobians with the key `True`.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        disciplines: Sequence[Discipline | EvaluationScenario],
        settings: BaseUMDOFormulationSettings | None = None,
        *,
        uncertain_space: RandomSpace,
        mdo_formulation_settings: BaseFormulationSettings | None = None,
    ) -> None:
        """
        Args:
            problem: The optimization problem
                defined over the design space,
                to which the statistics of the functions will be attached.
            uncertain_space: The uncertain variables
                with their probability distributions.
            mdo_formulation_settings: The settings of the MDO formulation
                generating the functions evaluable over the uncertain space.
                If `None`,
                use [MDF_Settings][gemseo.formulation.mdf_settings.MDF_Settings].
        """  # noqa: D205 D212 D415
        self.__available_statistics = self._statistic_factory.class_names
        self._uncertain_space = uncertain_space
        if mdo_formulation_settings is None:
            mdo_formulation_settings = MDF_Settings()

        mdo_formulation_class = mdo_formulation_factory.get_class(
            mdo_formulation_settings.target_class_name
        )

        # Update the design space as the MDO formulation would do it,
        # e.g. IDF adds the coupling variables to the design space.
        mdo_formulation_class(
            problem.__class__(problem.input_space),
            disciplines,
            settings=mdo_formulation_settings,
        )

        # Create the MDO formulation
        # whose functions are evaluable over the uncertain space
        # and differentiable with respect to the design variables.
        self._mdo_formulation = mdo_formulation_class(
            EvaluationProblem(uncertain_space),
            disciplines,
            settings=mdo_formulation_settings.model_copy(
                update={
                    "differentiated_input_names_substitute": list(
                        problem.input_space.variables
                    )
                }
            ),
        )

        if self._use_auxiliary_mdo_formulation:
            self._auxiliary_mdo_formulation = mdo_formulation_class(
                EvaluationProblem(uncertain_space),
                disciplines,
                settings=mdo_formulation_settings,
            )
        else:
            self._auxiliary_mdo_formulation = None

        super().__init__(problem, disciplines, settings=settings)
        self.name = (
            f"{self.__class__.__name__}[{self._mdo_formulation.__class__.__name__}]"
        )

        # Initialize the cache mechanism.
        self.input_data_to_output_data = {}
        self.problem.add_listener(self._clear_input_data_to_output_data)

    def _create_multidisciplinary_process(self) -> None:
        # The multidisciplinary processes are created by the MDO formulations.
        pass

    def _update_input_space(self) -> None:
        # The input space has been updated at instantiation by the MDO formulation.
        pass

    @staticmethod
    def __check_statistic_parameters(
        statistic_name: str,
        statistic_parameters: StrKeyMapping,
    ) -> None:
        """Check the statistic parameters.

        Args:
            statistic_name: The name of the statistic.
            statistic_parameters: The statistic parameters to be checked.

        Raises:
            ValueError: When the factor of a margin is zero.
        """
        if (
            statistic_name == _margin_name
            and statistic_parameters.get(_factor_name, _default_factor) == 0
        ):
            msg = (
                "The factor of a margin must be non-zero; "
                "a margin with a zero factor is the mean. "
                "Use the statistic 'Mean' instead."
            )
            raise ValueError(msg)

    @staticmethod
    def __update_statistic_parameters(
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
        if statistic_name == _margin_name:
            statistic_parameters = statistic_parameters or {}
            factor = abs(statistic_parameters.get(_factor_name, _default_factor))
            if use_negative_factor:
                factor *= -1
            statistic_parameters[_factor_name] = factor

        return statistic_parameters

    def _clear_input_data_to_output_data(self, x_vect: RealArray) -> None:
        """Clear the attribute `input_data_to_output_data` except for `x_vect`.

        This listener is called
        when the first output value at the new point `x_vect` is stored in the database,
        i.e. just after the estimation of a first statistic at `x_vect`:
        the data used for this estimation are kept for the other statistics.

        Args:
            x_vect: An input vector.
        """
        data = self.input_data_to_output_data
        current_output_data = data.get(key := HashableNdarray(x_vect))
        data.clear()
        if current_output_data is not None:
            data[key] = current_output_data

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

    def __add_mdo_observable(
        self,
        output_names: Sequence[str],
        observable_name: str = "",
        discipline: Discipline | None = None,
    ) -> str:
        """Add an observable to the MDO formulations if missing.

        Args:
            output_names: The names of the outputs to observe.
            observable_name: The name of the observable.
                If empty, concatenate the output names.
            discipline: The discipline computing the observed outputs.
                If `None`, the discipline is detected from inner disciplines.

        Returns:
            The name of the observable.
        """
        function_name = observable_name or "_".join(output_names)
        function_names = [
            function_.name
            for function_ in self._mdo_formulation.problem.functions
            if function_ is not None
        ]
        if function_name not in function_names:
            for formulation in (
                self._auxiliary_mdo_formulation,
                self._mdo_formulation,
            ):
                if formulation is not None:
                    formulation.add_observable(
                        output_names,
                        observable_name=observable_name,
                        discipline=discipline,
                    )

            self._post_add_mdo_observable()

        return function_name

    def create_objective(  # type: ignore[override]
        self,
        output_names: str | Iterable[str],
        statistic_name: str,
        objective_name: str = "",
        minimize: bool = True,
        **statistic_parameters: Any,
    ) -> ArrayFunction:
        """Create an objective function associated with outputs to be minimized.

        Args:
            output_names: The names of the outputs.
            statistic_name: The name of the statistic of `output_names`.
            objective_name: The name of the objective.
                If empty,
                the name is generated from `output_names` and `statistic_name`.
            minimize: Whether the statistic of the objective will be minimized.
            **statistic_parameters: The values of the parameters of the statistic,
                if any.

        Returns:
            The objective function.
        """
        self.__check_statistic_parameters(statistic_name, statistic_parameters)
        output_names = convert_strings_to_iterable(output_names)
        function_name = self.__add_mdo_observable(output_names)
        statistic_parameters = self.__update_statistic_parameters(
            statistic_name, statistic_parameters, not minimize
        )
        objective = self._statistic_function_class(
            self,
            function_name,
            ArrayFunction.FunctionType.OBJ,
            statistic_name,
            **statistic_parameters,
        )
        objective.name = objective_name or self.__compute_name(
            output_names, statistic_name, **statistic_parameters
        )
        return objective

    def add_observable(  # type: ignore[override]
        self,
        output_names: str | Sequence[str],
        statistic_name: str,
        observable_name: str = "",
        discipline: Discipline | None = None,
        **statistic_parameters: Any,
    ) -> None:
        """
        Args:
            statistic_name: The name of the statistic of `output_names`.
            **statistic_parameters: The values of the parameters of the statistic,
                if any.
        """  # noqa: D205 D212 D415
        self.__check_statistic_parameters(statistic_name, statistic_parameters)
        output_names = convert_strings_to_iterable(output_names)
        function_name = self.__add_mdo_observable(
            output_names, observable_name=observable_name, discipline=discipline
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

    def create_constraint(  # type: ignore[override]
        self,
        output_names: str | Sequence[str],
        statistic_name: str,
        constraint_type: ArrayFunction.ConstraintType = (
            ArrayFunction.ConstraintType.INEQ
        ),
        constraint_name: str = "",
        value: float = 0.0,
        positive: bool = False,
        **statistic_parameters: Any,
    ) -> ArrayFunction:
        """Create a constraint function associated with outputs names.

        Args:
            output_names: The names of the outputs.
            statistic_name: The name of the statistic of `output_names`.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored.
                If empty,
                the name is generated from `output_names` and `statistic_name`.
            value: The value $a$.
            positive: Whether the inequality constraint is positive.
            **statistic_parameters: The values of the parameters of the statistic,
                if any.

        Returns:
            The constraint function.
        """
        self.__check_statistic_parameters(statistic_name, statistic_parameters)
        output_names = convert_strings_to_iterable(output_names)
        function_name = self.__add_mdo_observable(output_names)
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
        name = self.__compute_name(output_names, statistic_name, **statistic_parameters)
        constraint.output_names = [name]
        if constraint_name:
            constraint.name = constraint_name
            constraint.has_default_name = False
        else:
            constraint.name = name
            constraint.has_default_name = True

        return Constraints.format(
            constraint,
            value=value,
            constraint_type=constraint_type,
            positive=positive,
        )

    def add_constraint(
        self,
        output_names: str | Sequence[str],
        statistic_name: str,
        constraint_type: ArrayFunction.ConstraintType = (
            ArrayFunction.ConstraintType.INEQ
        ),
        constraint_name: str = "",
        value: float = 0.0,
        positive: bool = False,
        **statistic_parameters: Any,
    ) -> None:
        """Add a constraint to the optimization problem.

        Args:
            output_names: The names of the outputs.
            statistic_name: The name of the statistic of `output_names`.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored.
                If empty,
                the name is generated from `output_names` and `statistic_name`.
            value: The value $a$.
            positive: Whether the inequality constraint is positive.
            **statistic_parameters: The values of the parameters of the statistic,
                if any.
        """
        self.problem.add_constraint(
            self.create_constraint(
                output_names,
                statistic_name,
                constraint_type=constraint_type,
                constraint_name=constraint_name,
                value=value,
                positive=positive,
                **statistic_parameters,
            )
        )
        self._post_add_constraint()

    def _post_add_mdo_observable(self) -> None:
        """Apply actions after adding an observable to the MDO formulations."""

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
            statistic_name: The name of the statistic of `output_name`.
            **statistic_parameters: The values of the parameters of the statistic,
                if any.

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

    def get_top_level_disciplines(  # noqa: D102
        self, include_sub_formulations: bool = False
    ) -> tuple[Discipline, ...]:
        return self._mdo_formulation.get_top_level_disciplines(
            include_sub_formulations=include_sub_formulations
        )
