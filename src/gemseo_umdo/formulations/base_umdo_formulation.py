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
from gemseo.optimization import OptimizationProblem
from gemseo.optimization.core.constraints import Constraints
from gemseo.space import DesignSpace
from gemseo.uncertainty.statistic.core.base import BaseStatistics
from gemseo.util.data_conversion import split_array_to_dict_of_arrays
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.hashable_ndarray import HashableNdarray
from gemseo.util.string import convert_strings_to_iterable
from gemseo.util.string import pretty_repr

from gemseo_umdo.formulations._functions.base_statistic_function import (
    BaseStatisticFunction,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import MutableMapping
    from collections.abc import Sequence

    from gemseo.core.base_factory import BaseFactory
    from gemseo.discipline import Discipline
    from gemseo.formulation.core.base_mdo import BaseMDOFormulation
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.scenario import EvaluationScenario
    from gemseo.space import RandomSpace
    from gemseo.util.typing import RealArray

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

    input_data_to_output_data: dict[
        HashableNdarray, tuple[dict[str, Any], dict[str, Any] | None]
    ]
    """The output samples or output statistics associated with the input data.

    For each input point,
    the data to estimate the statistics
    and the data to estimate their Jacobians, if computed, are stored as a tuple.
    Both are the same object when a single sampling computes them.
    """

    estimate_statistic_jacobians: bool
    """Whether to compute the data to estimate the Jacobians of the statistics
    when computing the data to estimate the statistics.

    When `True`,
    the disciplines are sampled once per design point
    to estimate both the statistics and their Jacobians,
    but they are also linearized at the design points
    where only the statistics are required,
    e.g. the line search points of a gradient-based optimizer.
    The U-scenarios set it before executing their driver,
    from the driver and the differentiation method of the problem,
    and reset it with `reset_execution_state()` at the end
    (see `_execute()` of
    [BaseUScenario][gemseo_umdo.scenarios.base_u_scenario.BaseUScenario]).

    In any case,
    it becomes `True` at the first request of the Jacobian of a statistic
    and stays `True` until it is reset,
    so that the next design points are sampled once for statistics and Jacobians.
    When it was `False` before this request,
    the design point of this request is sampled once more for the Jacobians,
    while the statistics at this point keep the values of the first sampling;
    the statistics and their Jacobians at this point
    are thus estimated from two different samplings.
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
        self.estimate_statistic_jacobians = False
        self.problem.add_listener(self._clear_input_data_to_output_data)

    def _create_multidisciplinary_process(self) -> None:
        # The multidisciplinary processes are created by the MDO formulations.
        pass

    def _update_input_space(self) -> None:
        # The input space has been updated at instantiation by the MDO formulation.
        pass

    def __update_statistic_parameters(
        self,
        statistic_name: str,
        statistic_parameters: MutableMapping[str, Any],
        conservative_from_below: bool | None,
    ) -> MutableMapping[str, Any]:
        """Update the statistic parameters with the class of its estimator.

        Args:
            statistic_name: The name of the statistic.
            statistic_parameters: The statistic parameters to be updated.
            conservative_from_below: Whether the statistic must be conservative
                from below,
                e.g. with a negative factor for a margin
                and a level lower than 0.5 for a quantile.
                If `None`,
                e.g. for an observable,
                the statistic does not have to be conservative.

        Returns:
            The updated statistic parameters.
        """
        statistic_class = self._statistic_factory.get_class(statistic_name)
        return statistic_class.update_parameters(
            statistic_parameters, conservative_from_below
        )

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
        current_key = HashableNdarray(x_vect)
        # The key of x_vect, if any, is kept as is, as it is already a copy.
        for key in [key for key in data if key != current_key]:
            del data[key]

    def _forget_input_point(self, input_data: RealArray) -> None:
        """Forget the state kept for an input point before estimating its statistics.

        It is called before a new sampling to estimate the statistics at this point,
        but not before a sampling to estimate only their Jacobians.
        By default, there is no such state.

        Args:
            input_data: The input point.
        """

    def reset_execution_state(self) -> None:
        """Reset the state of the formulation specific to an execution.

        The U-scenarios call it at the end of their execution,
        so that the evaluations of the functions after this execution,
        e.g. by a post-processing,
        do not linearize the disciplines unless their Jacobians are requested.
        """
        self.estimate_statistic_jacobians = False

    @property
    def _objective_and_constraint_functions(self) -> list[BaseStatisticFunction]:
        """The statistic functions of the objective and of the constraints.

        They are the statistic functions of the objective and of the constraints
        of the problem,
        before any transformation hiding their type,
        e.g. the negation of an objective to maximize or of a positive constraint.
        A problem without objective and constraints,
        e.g. the evaluation problem of a U-evaluation scenario,
        has none.
        """
        problem = self.problem
        if not isinstance(problem, OptimizationProblem):
            return []

        # The original functions are the functions before any preprocessing.
        functions = list(problem.constraints.get_originals())
        if problem.objective is not None:
            functions.append(problem.objective.original)

        return BaseStatisticFunction.get_statistic_functions(functions)

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
        self._check_statistic_name(statistic_name)
        output_names = convert_strings_to_iterable(output_names)
        statistic_parameters = self.__update_statistic_parameters(
            statistic_name, statistic_parameters, not minimize
        )
        # The statistic function is created before adding the MDO observable
        # so that invalid statistic parameters do not leave an orphan observable.
        objective = self._statistic_function_class(
            self,
            self.__get_mdo_observable_name(output_names),
            ArrayFunction.FunctionType.OBJ,
            statistic_name,
            **statistic_parameters,
        )
        if self.__check_mdo_observables(output_names):
            self.__add_mdo_observables(output_names)

        objective.name = objective_name or self.__compute_name(
            output_names, statistic_name, **statistic_parameters
        )
        self._register_statistic_function(objective)
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
        self._check_statistic_name(statistic_name)
        output_names = convert_strings_to_iterable(output_names)
        statistic_parameters = self.__update_statistic_parameters(
            statistic_name, statistic_parameters, None
        )
        observable = self._statistic_function_class(
            self,
            self.__get_mdo_observable_name(output_names, observable_name),
            ArrayFunction.FunctionType.NONE,
            statistic_name,
            **statistic_parameters,
        )
        observable.name = self.__compute_name(
            observable_name or output_names, statistic_name, **statistic_parameters
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
        output_names = convert_strings_to_iterable(output_names)
        function, constraint = self.__create_constraint(
            output_names,
            statistic_name,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
            **statistic_parameters,
        )
        if self.__check_mdo_observables(output_names):
            self.__add_mdo_observables(output_names)

        self._register_statistic_function(function)
        return constraint

    def __create_constraint(
        self,
        output_names: Sequence[str],
        statistic_name: str,
        constraint_type: ArrayFunction.ConstraintType,
        constraint_name: str,
        value: float,
        positive: bool,
        **statistic_parameters: Any,
    ) -> tuple[BaseStatisticFunction, ArrayFunction]:
        """Create a constraint function associated with outputs names.

        The MDO observable computing the outputs is not added.

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
            The function computing the statistic
            and the constraint function built from it.
        """
        self._check_statistic_name(statistic_name)
        # An equality constraint does not have to be conservative.
        statistic_parameters = self.__update_statistic_parameters(
            statistic_name,
            statistic_parameters,
            positive if constraint_type == ArrayFunction.ConstraintType.INEQ else None,
        )
        constraint = self._statistic_function_class(
            self,
            self.__get_mdo_observable_name(output_names),
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

        formatted_constraint = Constraints.format(
            constraint,
            value=value,
            constraint_type=constraint_type,
            positive=positive,
        )
        if formatted_constraint is not constraint:
            # The formatted constraint is a transformation hiding its type,
            # e.g. the negation of a positive constraint.
            constraint.mark_transformation(formatted_constraint)

        return constraint, formatted_constraint

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
        output_names = convert_strings_to_iterable(output_names)
        function, constraint = self.__create_constraint(
            output_names,
            statistic_name,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
            **statistic_parameters,
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
