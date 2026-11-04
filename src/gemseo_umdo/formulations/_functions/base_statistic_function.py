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
"""A function to compute a statistic from a `BaseUMDOFormulation`.

See Also:
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation].
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Generic
from typing import TypeVar

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.problem.database import Database
from gemseo.uncertainty.statistic.core.base import BaseStatistics
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.hashable_ndarray import HashableNdarray
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.doe import EvaluationProblem
    from gemseo.util.typing import RealArray

    from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
        BaseStatisticEstimator,
    )
    from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation

UMDOFormulationT = TypeVar("UMDOFormulationT", bound="BaseUMDOFormulation")

_statistic_function_attribute_name: Final[str] = "_umdo_statistic_function"
"""The name of the attribute of a transformed statistic function storing it.

E.g. a constraint with an offset.
The negation of a function does not need it,
as it evaluates a method of this function.
"""


class BaseStatisticFunction(
    ArrayFunction,
    Generic[UMDOFormulationT],
    metaclass=ABCGoogleDocstringInheritanceMeta,
):
    """A function to estimate a statistic from a U-MDO formulation."""

    _can_estimate_jacobian: ClassVar[bool] = False
    """Whether the data computed by this class allow to estimate the Jacobian.

    Whether the Jacobian of a given statistic can be estimated
    also depends on its estimator.
    """

    _output_jac_name: str
    """The name of the Jacobian of the output of interest."""

    _output_name: str
    """The name of the output of interest."""

    _statistic_estimator: BaseStatisticEstimator
    """The statistic estimator."""

    _statistic_jac_name: str
    """The name of the Jacobian of the statistic of the output of interest."""

    _statistic_name: str
    """The name of the statistic of the output of interest."""

    _umdo_formulation: UMDOFormulationT
    """The U-MDO formulation associated with this function."""

    def __init__(
        self,
        umdo_formulation: UMDOFormulationT,
        output_name: str,
        function_type: ArrayFunction.FunctionType,
        statistic_operator_name: str,
        **statistic_options: Any,
    ) -> None:
        """
        Args:
            umdo_formulation: The U-MDO formulation
                associated with the output of interest.
            output_name: The name of the output of interest.
            function_type: The type of function.
            statistic_operator_name: The name of the statistic operator.
            **statistic_options: The options of the statistic.
        """  # noqa: D205 D212 D415
        self._output_name = output_name
        self._output_jac_name = Database.get_gradient_name(output_name)
        self._umdo_formulation = umdo_formulation
        self._statistic_estimator = umdo_formulation._statistic_factory.create(
            statistic_operator_name,
            *self._statistic_estimator_parameters,
            **statistic_options,
        )
        # The statistic options are part of the name
        # so that two statistics differing only in their options,
        # e.g. two margins with different factors, have different names.
        self._statistic_name = BaseStatistics.compute_expression(
            output_name,
            FilePathManager.to_snake_case(statistic_operator_name),
            show_name=True,
            **statistic_options,
        )
        self._statistic_jac_name = Database.get_gradient_name(self._statistic_name)
        super().__init__(
            self._func, name=output_name, f_type=function_type, jac=self._jac
        )

    @property
    def _statistic_estimator_parameters(self) -> tuple[Any, ...]:
        """The parameters of the estimator of the statistic."""
        return ()

    def __compute_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> RealArray:
        """Estimate the statistic or its Jacobian at an input point.

        Args:
            input_data: The input point.
            estimate_jacobian: Whether to estimate the Jacobian of the statistic
                with respect to the input variables.

        Returns:
            The estimation of the statistic or its Jacobian at the given input point.
        """
        umdo_formulation = self._umdo_formulation
        # From the first Jacobian request, sample once for statistics and Jacobians.
        umdo_formulation.estimate_statistic_jacobians |= estimate_jacobian

        hashable_input_data = HashableNdarray(input_data)
        i_to_o = umdo_formulation.input_data_to_output_data

        # 1. We test whether the data to estimate the statistic or its Jacobian
        #    have already been computed at this input point,
        #    e.g. when estimating another statistic.
        data_for_statistic, data_for_jacobian = i_to_o.get(
            hashable_input_data, (None, None)
        )
        data_for_statistic_estimation = (
            data_for_jacobian if estimate_jacobian else data_for_statistic
        )
        if data_for_statistic_estimation is None:
            # The key is a copy of the input point
            # so that a change of the input array during or after this call,
            # e.g. by a driver,
            # does not change the key.
            hashable_input_data.copy_wrapped_array()
            # 2. If not, we compute them from a new sampling,
            #    so we reset the evaluation sub-problems
            #    to forget the samples of the previous sampling,
            #    whose size may differ, e.g. with SequentialSampling.
            problems = [umdo_formulation.mdo_formulation.problem]
            formulation = umdo_formulation.auxiliary_mdo_formulation
            if formulation is not None:
                problems.append(formulation.problem)
            problems.extend(self._other_evaluation_problems)
            for problem in problems:
                problem.reset()

            if not estimate_jacobian:
                # The statistics at this input point are estimated again.
                umdo_formulation._forget_input_point(input_data)

            umdo_formulation.update_top_level_disciplines(input_data)
            # The data to estimate the Jacobian also allow to estimate the statistic,
            # so the disciplines are sampled once per input point
            # when the Jacobians are required.
            with_jacobian = umdo_formulation.estimate_statistic_jacobians
            data_for_statistic_estimation = self._compute_data_for_statistic_estimation(
                input_data, with_jacobian
            )
            if data_for_statistic is None:
                data_for_statistic = data_for_statistic_estimation

            # The statistics already estimated at this input point
            # keep their values when it is sampled again for the Jacobians.
            i_to_o[hashable_input_data] = (
                data_for_statistic,
                data_for_statistic_estimation if with_jacobian else None,
            )

        # 3. We compute the estimation of the statistic of interest or its Jacobian.
        if estimate_jacobian:
            compute_estimation = self._compute_statistic_jacobian_estimation
        else:
            compute_estimation = self._compute_statistic_estimation
        return compute_estimation(data_for_statistic_estimation)

    def _func(self, input_data: RealArray) -> RealArray:
        """A function estimating the statistic at an input point.

        Args:
            input_data: The input point.

        Returns:
            The estimation of the statistic at the given input point.
        """
        return self.__compute_statistic_estimation(input_data, False)

    def _jac(self, input_data: RealArray) -> RealArray:
        """A function estimating the Jacobian of the statistic at an input point.

        Args:
            input_data: The input point.

        Returns:
            The estimation of the Jacobian of the statistic at the given input point.

        Raises:
            NotImplementedError: When the Jacobian of the statistic
                cannot be estimated.
        """
        # The error is raised before any sampling
        # and before enabling the joint sampling of statistics and Jacobians.
        if not self.can_estimate_jacobian:
            msg = (
                "The Jacobian of the statistic "
                f"{self._statistic_estimator.__class__.__name__} "
                f"of {self._output_name} cannot be estimated "
                f"with {self._umdo_formulation.__class__.__name__}."
            )
            raise NotImplementedError(msg)

        return self.__compute_statistic_estimation(input_data, True)

    def mark_transformation(self, function: ArrayFunction) -> None:
        """Store this statistic function in a transformation of it.

        Args:
            function: The transformation of this statistic function
                that does not evaluate one of its methods,
                e.g. its offset.
        """
        setattr(function, _statistic_function_attribute_name, self)

    @classmethod
    def get_statistic_function(
        cls, function: ArrayFunction
    ) -> BaseStatisticFunction | None:
        """Return the statistic function of a function.

        Args:
            function: Either a statistic function,
                a transformation of a statistic function
                marked with `mark_transformation()`
                or a negation of such functions,
                e.g. the objective to maximize
                or the objective after toggling `minimize_objective` twice.

        Returns:
            The statistic function if any, otherwise `None`.
        """
        while True:
            if isinstance(function, BaseStatisticFunction):
                return function

            statistic_function = getattr(
                function, _statistic_function_attribute_name, None
            )
            if statistic_function is not None:
                return statistic_function

            # The negation of a function evaluates a method of this function,
            # so this function is found from the bound method.
            operand = getattr(function.func, "__self__", None)
            if not isinstance(operand, ArrayFunction) or operand is function:
                return None

            function = operand

    @classmethod
    def get_statistic_functions(
        cls, functions: Iterable[ArrayFunction]
    ) -> list[BaseStatisticFunction]:
        """Return the statistic functions of functions.

        Args:
            functions: The functions,
                either statistic functions or transformations of statistic functions,
                e.g. the negation of an objective to maximize,
                or other functions.

        Returns:
            The statistic functions of the functions that have one.
        """
        return [
            statistic_function
            for function in functions
            if (statistic_function := cls.get_statistic_function(function)) is not None
        ]

    @property
    def can_estimate_jacobian(self) -> bool:
        """Whether the Jacobian of the statistic can be estimated."""
        estimator = self._statistic_estimator
        return self._can_estimate_jacobian and estimator.can_compute_jacobian

    @abstractmethod
    def _compute_statistic_estimation(self, data: dict[str, Any]) -> RealArray:
        """Estimate the statistic.

        Args:
            data: The data to estimate the Jacobian of the statistic.

        Returns:
            The estimation of the statistic.
        """

    def _compute_statistic_jacobian_estimation(self, data: dict[str, Any]) -> RealArray:
        """Estimate the Jacobian of the statistic.

        Args:
            data: The data to estimate the Jacobian of the statistic.

        Returns:
            The estimation of the Jacobian of the statistic.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, Any]:
        """Compute the data to estimate the statistic or its Jacobian at an input point.

        Args:
            input_data: The input point.
            estimate_jacobian: Whether to estimate the Jacobian of the statistic
                with respect to the input variables.
                In this case,
                the data must also allow to estimate the statistic.

        Returns:
            The data to estimate the statistic or its Jacobian at the given input point.
        """

    @property
    def _other_evaluation_problems(self) -> tuple[EvaluationProblem, ...]:
        """Any evaluation problem different from the two main.

        The two main problems defined in the U-MDO formulation are:

        - the one
          that evaluates the functions over the uncertain space
          and differentiates with respect to the design variables
          (see :attr:`.BaseUMDOFormulation._mdo_formulation`),
        - the one
          that evaluates the functions over the uncertain space
          and differentiates with respect to the uncertain variables
          (see :attr:`.BaseUMDOFormulation._auxiliary_mdo_formulation`).
          This latter problem is only defined when required,
          depending on the formulation.
        """
        return ()
