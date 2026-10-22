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

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.problem.database import Database
from gemseo.uncertainty.statistic.core.base import BaseStatistics
from gemseo.util.file_path_manager import FilePathManager
from gemseo.util.hashable_ndarray import HashableNdarray
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.doe import EvaluationProblem
    from gemseo.util.typing import RealArray

    from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
        BaseStatisticEstimator,
    )
    from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation

UMDOFormulationT = TypeVar("UMDOFormulationT", bound="BaseUMDOFormulation")


class BaseStatisticFunction(
    ArrayFunction,
    Generic[UMDOFormulationT],
    metaclass=ABCGoogleDocstringInheritanceMeta,
):
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
        hashable_input_data = HashableNdarray(input_data)
        i_to_o = umdo_formulation.input_data_to_output_data

        # 1. We test whether the data to estimate the statistic or its Jacobian
        #    have already been computed at this input point,
        #    e.g. when estimating another statistic.
        input_data_to_data = i_to_o.setdefault(hashable_input_data, {})
        data_for_statistic_estimation = input_data_to_data.get(estimate_jacobian)
        if data_for_statistic_estimation is None:
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

            umdo_formulation.update_top_level_disciplines(input_data)
            data_for_statistic_estimation = self._compute_data_for_statistic_estimation(
                input_data, estimate_jacobian
            )
            input_data_to_data[estimate_jacobian] = data_for_statistic_estimation

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
