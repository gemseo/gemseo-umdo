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
from gemseo.util.string import pretty_repr

from gemseo_umdo.formulations.statistic_settings import BaseStatisticSettings

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

    from gemseo_umdo.formulations._functions.base_statistic_function import (
        BaseStatisticFunction,
    )
    from gemseo_umdo.formulations.base_umdo_formulation_settings import (
        BaseUMDOFormulationSettings,
    )


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
    """A factory of statistic estimators.

    A U-MDO formulation with only one way to estimate statistics sets it at class level.
    A U-MDO formulation with several ways sets it at instantiation,
    before calling the `__init__` of this class.
    For example, `Sampling` can estimate statistics in one go or iteratively.
    """

    _statistic_function_class: type[BaseStatisticFunction]
    """A subclass of `ArrayFunction` to compute a statistic, set at class level."""

    _uncertain_space: RandomSpace
    """The uncertain space."""

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

    def __get_statistic_settings(
        self,
        statistic_settings: str | BaseStatisticSettings,
        conservative_from_below: bool | None,
    ) -> BaseStatisticSettings:
        """Return the settings of a statistic, checked and made conservative.

        Args:
            statistic_settings: Either the settings of the statistic
                or the name of the statistic to use its default settings.
            conservative_from_below: Whether the statistic must be conservative
                from below,
                e.g. with a negative factor for a margin
                and a level lower than 0.5 for a quantile.
                If `None`,
                e.g. for an observable,
                the statistic does not have to be conservative.

        Returns:
            The settings of the statistic.
        """
        if isinstance(statistic_settings, str):
            # The availability is checked first
            # so that the error lists the statistics available for this formulation.
            self._check_statistic_name(statistic_settings)
            statistic_settings = BaseStatisticSettings.from_name(statistic_settings)
        else:
            self._check_statistic_name(statistic_settings.target_class_name)
            # The estimator reads its settings at each estimation,
            # so it uses a copy that the user cannot change afterwards.
            statistic_settings = statistic_settings.model_copy()

        if conservative_from_below is None:
            return statistic_settings

        return statistic_settings.make_conservative(conservative_from_below)

    def _register_statistic_function(self, function: BaseStatisticFunction) -> None:
        """Register a function computing a statistic.

        This method is called once the function has been created
        by `create_objective` or `create_constraint`
        or added to the optimization problem
        by `add_constraint` or `add_observable`,
        so that a failure leaves the U-MDO formulation unchanged.

        Args:
            function: The function computing the statistic.
        """

    def _check_statistic_name(self, statistic_name: str) -> None:
        """Check that a statistic is available.

        Args:
            statistic_name: The name of the statistic.

        Raises:
            ValueError: When the statistic is not available.
        """
        if not self._statistic_factory.is_available(statistic_name):
            msg = (
                f"The statistic '{statistic_name}' is not available "
                f"for the U-MDO formulation {self.__class__.__name__}; "
                f"available statistics: {pretty_repr(self.available_statistics)}."
            )
            raise ValueError(msg)

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
        return self._statistic_factory.class_names

    @staticmethod
    def __get_mdo_observable_name(
        output_names: Sequence[str], observable_name: str = ""
    ) -> str:
        """Return the name of the observable of the MDO formulations.

        Args:
            output_names: The names of the outputs to observe.
            observable_name: The name of the observable.
                If empty, concatenate the output names.

        Returns:
            The name of the observable.
        """
        return observable_name or "_".join(output_names)

    def __check_mdo_observables(
        self,
        output_names: Sequence[str],
        observable_name: str = "",
        discipline: Discipline | None = None,
    ) -> bool:
        """Check the observables of the MDO formulations to be added, if missing.

        The observables are added by `__add_mdo_observables()`;
        checking them first rejects wrong output names
        before changing the optimization problem.

        Args:
            output_names: The names of the outputs to observe.
            observable_name: The name of the observable.
                If empty, concatenate the output names.
            discipline: The discipline computing the observed outputs.
                If `None`, the discipline is detected from inner disciplines.

        Returns:
            Whether the observable is missing.
        """
        function_name = self.__get_mdo_observable_name(output_names, observable_name)
        function_names = [
            function_.name
            for function_ in self._mdo_formulation.problem.functions
            if function_ is not None
        ]
        if function_name in function_names:
            return False

        for formulation in (self._auxiliary_mdo_formulation, self._mdo_formulation):
            if formulation is None:
                continue

            disciplines = (
                formulation.get_top_level_disciplines()
                if discipline is None
                else (discipline,)
            )
            if not any(
                discipline_.io.output_grammar.has_names(output_names)
                for discipline_ in disciplines
            ):
                # Creating the function raises the error for a wrong output name;
                # it is not created otherwise as add_observable() will create it.
                formulation.create_function(
                    output_names, discipline=discipline, name=observable_name
                )

        return True

    def __add_mdo_observables(
        self,
        output_names: Sequence[str],
        observable_name: str = "",
        discipline: Discipline | None = None,
    ) -> None:
        """Add observables to the MDO formulations.

        Args:
            output_names: The names of the outputs to observe.
            observable_name: The name of the observable.
                If empty, concatenate the output names.
            discipline: The discipline computing the observed outputs.
                If `None`, the discipline is detected from inner disciplines.
        """
        for formulation in (self._auxiliary_mdo_formulation, self._mdo_formulation):
            if formulation is not None:
                formulation.add_observable(
                    output_names=output_names,
                    observable_name=observable_name,
                    discipline=discipline,
                )

        self._post_add_mdo_observable()

    def create_objective(  # type: ignore[override]
        self,
        output_names: str | Iterable[str],
        statistic_settings: str | BaseStatisticSettings,
        objective_name: str = "",
        minimize: bool = True,
    ) -> ArrayFunction:
        """Create an objective function associated with outputs to be minimized.

        Args:
            output_names: The names of the outputs.
            statistic_settings: Either the settings of the statistic of `output_names`
                or the name of the statistic to use its default settings,
                e.g. `"Mean"`.
            objective_name: The name of the objective.
                If empty,
                the name is generated from `output_names` and `statistic_settings`.
            minimize: Whether the statistic of the objective will be minimized.

        Returns:
            The objective function.
        """
        output_names = convert_strings_to_iterable(output_names)
        statistic_settings = self.__get_statistic_settings(
            statistic_settings, not minimize
        )
        # The statistic function is created before adding the MDO observable
        # so that an invalid statistic does not leave an orphan observable.
        objective = self._statistic_function_class(
            self,
            self.__get_mdo_observable_name(output_names),
            ArrayFunction.FunctionType.OBJ,
            statistic_settings,
        )
        if self.__check_mdo_observables(output_names):
            self.__add_mdo_observables(output_names)

        objective.name = objective_name or self.__compute_name(
            output_names, statistic_settings
        )
        self._register_statistic_function(objective)
        return objective

    def add_observable(  # type: ignore[override]
        self,
        output_names: str | Sequence[str],
        statistic_settings: str | BaseStatisticSettings,
        observable_name: str = "",
        discipline: Discipline | None = None,
    ) -> None:
        """
        Args:
            statistic_settings: Either the settings of the statistic of `output_names`
                or the name of the statistic to use its default settings,
                e.g. `"Mean"`.
        """  # noqa: D205 D212 D415
        output_names = convert_strings_to_iterable(output_names)
        statistic_settings = self.__get_statistic_settings(statistic_settings, None)
        observable = self._statistic_function_class(
            self,
            self.__get_mdo_observable_name(output_names, observable_name),
            ArrayFunction.FunctionType.NONE,
            statistic_settings,
        )
        observable.name = self.__compute_name(
            observable_name or output_names, statistic_settings
        )
        # The observable is added to the problem before the MDO observables,
        # so that a failure, e.g. a name collision,
        # does not leave an orphan observable.
        is_mdo_observable_missing = self.__check_mdo_observables(
            output_names, observable_name=observable_name, discipline=discipline
        )
        is_observed = observable.name in self.problem.observables.get_names()
        # The problem logs a warning and ignores the observable if already observed.
        self.problem.add_observable(observable)
        if is_observed:
            return

        if is_mdo_observable_missing:
            self.__add_mdo_observables(
                output_names, observable_name=observable_name, discipline=discipline
            )

        self._post_add_observable()
        self._register_statistic_function(observable)

    def create_constraint(  # type: ignore[override]
        self,
        output_names: str | Sequence[str],
        statistic_settings: str | BaseStatisticSettings,
        constraint_type: ArrayFunction.ConstraintType = (
            ArrayFunction.ConstraintType.INEQ
        ),
        constraint_name: str = "",
        value: float = 0.0,
        positive: bool = False,
    ) -> ArrayFunction:
        """Create a constraint function associated with outputs names.

        Args:
            output_names: The names of the outputs.
            statistic_settings: Either the settings of the statistic of `output_names`
                or the name of the statistic to use its default settings,
                e.g. `"Mean"`.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored.
                If empty,
                the name is generated from `output_names` and `statistic_settings`.
            value: The value $a$.
            positive: Whether the inequality constraint is positive.

        Returns:
            The constraint function.
        """
        output_names = convert_strings_to_iterable(output_names)
        function, constraint = self.__create_constraint(
            output_names,
            statistic_settings,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
        )
        if self.__check_mdo_observables(output_names):
            self.__add_mdo_observables(output_names)

        self._register_statistic_function(function)
        return constraint

    def __create_constraint(
        self,
        output_names: Sequence[str],
        statistic_settings: str | BaseStatisticSettings,
        constraint_type: ArrayFunction.ConstraintType,
        constraint_name: str,
        value: float,
        positive: bool,
    ) -> tuple[BaseStatisticFunction, ArrayFunction]:
        """Create a constraint function associated with outputs names.

        The MDO observable computing the outputs is not added.

        Args:
            output_names: The names of the outputs.
            statistic_settings: Either the settings of the statistic of `output_names`
                or the name of the statistic to use its default settings,
                e.g. `"Mean"`.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored.
                If empty,
                the name is generated from `output_names` and `statistic_settings`.
            value: The value $a$.
            positive: Whether the inequality constraint is positive.

        Returns:
            The function computing the statistic
            and the constraint function built from it.
        """
        # An equality constraint does not have to be conservative.
        statistic_settings = self.__get_statistic_settings(
            statistic_settings,
            positive if constraint_type == ArrayFunction.ConstraintType.INEQ else None,
        )
        constraint = self._statistic_function_class(
            self,
            self.__get_mdo_observable_name(output_names),
            ArrayFunction.FunctionType.NONE,
            statistic_settings,
        )
        name = self.__compute_name(output_names, statistic_settings)
        constraint.output_names = [name]
        if constraint_name:
            constraint.name = constraint_name
            constraint.has_default_name = False
        else:
            constraint.name = name
            constraint.has_default_name = True

        return (
            constraint,
            Constraints.format(
                constraint,
                value=value,
                constraint_type=constraint_type,
                positive=positive,
            ),
        )

    def add_constraint(
        self,
        output_names: str | Sequence[str],
        statistic_settings: str | BaseStatisticSettings,
        constraint_type: ArrayFunction.ConstraintType = (
            ArrayFunction.ConstraintType.INEQ
        ),
        constraint_name: str = "",
        value: float = 0.0,
        positive: bool = False,
    ) -> None:
        """Add a constraint to the optimization problem.

        Args:
            output_names: The names of the outputs.
            statistic_settings: Either the settings of the statistic of `output_names`
                or the name of the statistic to use its default settings,
                e.g. `"Mean"`.
            constraint_type: The type of constraint.
            constraint_name: The name of the constraint to be stored.
                If empty,
                the name is generated from `output_names` and `statistic_settings`.
            value: The value $a$.
            positive: Whether the inequality constraint is positive.
        """
        output_names = convert_strings_to_iterable(output_names)
        function, constraint = self.__create_constraint(
            output_names,
            statistic_settings,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
        )
        # The constraint is added to the problem before the MDO observables,
        # so that a failure, e.g. a name collision,
        # does not leave an orphan observable.
        is_mdo_observable_missing = self.__check_mdo_observables(output_names)
        self.problem.add_constraint(constraint)
        if is_mdo_observable_missing:
            self.__add_mdo_observables(output_names)

        self._post_add_constraint()
        self._register_statistic_function(function)

    def _post_add_mdo_observable(self) -> None:
        """Apply actions after adding an observable to the MDO formulations."""

    def _post_add_constraint(self) -> None:
        """Apply actions after adding a constraint."""

    def _post_add_observable(self) -> None:
        """Apply actions after adding an observable."""

    @staticmethod
    def __compute_name(
        output_name: str | Iterable[str],
        statistic_settings: BaseStatisticSettings,
    ) -> str:
        """Create the string representation of a statistic applied to output variables.

        Args:
            output_name: Either the names of the output variables
                for which to estimate the statistic or a unique name to define them.
            statistic_settings: The settings of the statistic of `output_name`.

        Returns:
            The string representations of the statistic applied to the output variables.
        """
        if not isinstance(output_name, str):
            output_name = "_".join(output_name)

        return BaseStatistics.compute_expression(
            output_name,
            FilePathManager.to_snake_case(statistic_settings.target_class_name),
            **statistic_settings.get_name_parameters(),
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
