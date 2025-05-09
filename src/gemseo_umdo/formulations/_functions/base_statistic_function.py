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
from typing import Generic
from typing import TypeVar

from gemseo.algos.database import Database
from gemseo.algos.hashable_ndarray import HashableNdarray
from gemseo.core.mdo_functions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.algos.evaluation_problem import EvaluationProblem
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
        BaseStatisticEstimator,
    )
    from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation

UMDOFormulationT = TypeVar("UMDOFormulationT", bound="BaseUMDOFormulation")


class BaseStatisticFunction(MDOFunction, Generic[UMDOFormulationT]):
    """A function to estimate a statistic from a U-MDO formulation."""

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
        function_type: MDOFunction.FunctionType,
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
        self._statistic_name = (
            f"{self._statistic_estimator.__class__.__name__}[{output_name}]"
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
        hashable_input_data = HashableNdarray(input_data)
        i_to_o = umdo_formulation.input_data_to_output_data
        output_name = self._output_jac_name if estimate_jacobian else self._output_name

        # 1. We test whether input_data is a new input point.
        last_input_data = next(reversed(i_to_o)) if i_to_o else None
        if hashable_input_data != last_input_data:
            # This is a new design point, so we reset the evaluation sub-problems.
            problems = [umdo_formulation.mdo_formulation.optimization_problem]
            if (formulation := umdo_formulation.auxiliary_mdo_formulation) is not None:
                problems.append(formulation.optimization_problem)
            problems.extend(self._other_evaluation_problems)
            for problem in problems:
                problem.reset(preprocessing=False)

        # 2. We test whether the output has already been evaluated at this input point.
        #    Note: output means output f or its derivative @f.
        data_for_statistic_estimation = i_to_o.get(hashable_input_data, {})
        if output_name not in data_for_statistic_estimation:
            # This output has not yet been evaluated,
            # e.g. an output f has been evaluated
            # when calling __compute_statistic_estimation via _func
            # but not its Jacobian @f
            # to be evaluated in the current call to __compute_statistic_estimation.
            umdo_formulation.update_top_level_disciplines(input_data)
            data_for_statistic_estimation = self._compute_data_for_statistic_estimation(
                input_data, estimate_jacobian
            )
            i_to_o[hashable_input_data] = data_for_statistic_estimation

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
        """
        return self.__compute_statistic_estimation(input_data, True)

    @abstractmethod
    def _compute_statistic_estimation(self, data: dict[str, Any]) -> RealArray:
        """Estimate the statistic.

        Args:
            data: The data to estimate the Jacobian of the statistic.

        Returns:
            The estimation of the statistic.
        """

    @abstractmethod
    def _compute_statistic_jacobian_estimation(self, data: dict[str, Any]) -> RealArray:
        """Estimate the Jacobian of the statistic.

        Args:
            data: The data to estimate the Jacobian of the statistic.

        Returns:
            The estimation of the Jacobian of the statistic.
        """

    @abstractmethod
    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, Any]:
        """Compute the data to estimate the statistic or its Jacobian at an input point.

        Args:
            input_data: The input point.
            estimate_jacobian: Whether to estimate the Jacobian of the statistic
                with respect to the input variables.

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
