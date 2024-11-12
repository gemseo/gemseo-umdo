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
    """A function to compute a statistic from a `BaseUMDOFormulation`."""

    _statistic_estimator: BaseStatisticEstimator
    """A callable to estimate the statistic."""

    _umdo_formulation: UMDOFormulationT
    """The U-MDO formulation to which the
    [BaseStatisticFunction][gemseo_umdo.formulations.f
    unctions.base_statistic_function.BaseStatisticFunction] is attached."""

    _function_name: str
    """The name of the function."""

    _function_jac_name: str
    """The name of the Jacobian function."""

    _observable_name: str
    """The name of the observable that corresponds to the statistic of the function."""

    _observable_jac_name: str
    """The name of the Jacobian of the observable."""

    _last_input_data: HashableNdarray
    """The last input data passed to `__compute_qoi`."""

    def __init__(
        self,
        umdo_formulation: UMDOFormulationT,
        function: MDOFunction,
        function_type: MDOFunction.FunctionType,
        name: str,
        **statistic_options: Any,
    ) -> None:
        """
        Args:
            umdo_formulation: The U-MDO formulation
                to which the
                [BaseStatisticFunction][gemseo_umdo.formulations._functions.base_statistic_function.BaseStatisticFunction]
                is attached.
            function: The function for which we want to estimate an output statistic.
            function_type: The type of function.
            name: The name of the statistic.
            **statistic_options: The options of the statistic.
        """  # noqa: D205 D212 D415
        function_name = function.name
        self._function_name = function_name
        self._function_jac_name = Database.get_gradient_name(function_name)
        self._umdo_formulation = umdo_formulation
        self._last_input_data = None
        self._statistic_estimator = umdo_formulation._statistic_factory.create(
            name, *self._statistic_estimator_parameters, **statistic_options
        )
        self._observable_name = (
            f"{self._statistic_estimator.__class__.__name__}[{function_name}]"
        )
        self._observable_jac_name = Database.get_gradient_name(self._observable_name)
        super().__init__(
            self._func, name=function_name, f_type=function_type, jac=self._jac
        )

    @property
    def _statistic_estimator_parameters(self) -> tuple[Any, ...]:
        """The parameters of the estimator of the statistic."""
        return ()

    def __compute_qoi(self, input_data: RealArray, is_jac: bool = False) -> RealArray:
        """A function computing a quantity of interest (QOI) at a given point.

        Args:
            input_data: The input point (values of design variables)
                  at which to estimate the statistic.
            is_jac: Whether the quantity of interest is a Jacobian.

        Returns:
            The quantity of interest at the given input point.
        """
        umdo_formulation = self._umdo_formulation
        hashable_input_data = HashableNdarray(input_data)
        if hashable_input_data != self._last_input_data:
            self._last_input_data = hashable_input_data

            problems = [umdo_formulation.mdo_formulation.optimization_problem]

            auxiliary_mdo_formulation = umdo_formulation.auxiliary_mdo_formulation
            if auxiliary_mdo_formulation is not None:
                problems.append(auxiliary_mdo_formulation.optimization_problem)

            problems.extend(self._other_evaluation_problems)

            for problem in problems:
                problem.reset(preprocessing=False)

        input_data_to_output_data = umdo_formulation.input_data_to_output_data
        output_data = input_data_to_output_data.get(hashable_input_data, {})
        name = self._function_jac_name if is_jac else self._function_name
        if name not in output_data:
            umdo_formulation.update_top_level_disciplines(input_data)
            output_data = input_data_to_output_data[hashable_input_data] = {}
            self._compute_output_data(
                input_data,
                output_data,
                compute_jacobian=is_jac,
            )

        if is_jac:
            return self._compute_statistic_jacobian_estimation(output_data)

        return self._compute_statistic_estimation(output_data)

    def _func(self, input_data: RealArray) -> RealArray:
        """A function estimating the statistic at a given input point.

        Args:
            input_data: The input point at which to estimate the statistic.

        Returns:
            The estimation of the statistic at the given input point.
        """
        return self.__compute_qoi(input_data)

    def _jac(self, input_data: RealArray) -> RealArray:
        """A function estimating the Jacobian of the statistic at a given input point.

        Args:
            input_data: The input point at which to estimate the statistic.

        Returns:
            The estimation of the Jacobian of the statistic at the given input point.
        """
        return self.__compute_qoi(input_data, is_jac=True)

    @abstractmethod
    def _compute_statistic_estimation(self, data: dict[str, Any]) -> RealArray:
        """Estimate the statistic.

        Args:
            data: The data from which to estimate the statistic.

        Returns:
            The estimation of the statistic.
        """

    @abstractmethod
    def _compute_statistic_jacobian_estimation(self, data: dict[str, Any]) -> RealArray:
        """Estimate the Jacobian of the statistic.

        Args:
            data: The data from which to estimate the Jacobian of the statistic.

        Returns:
            The estimation of the Jacobian of the statistic.
        """

    @abstractmethod
    def _compute_output_data(
        self,
        input_data: RealArray,
        output_data: dict[str, Any],
        compute_jacobian: bool = False,
    ) -> None:
        """Compute the output data.

        Args:
            input_data: The input point at which to estimate the statistic.
            output_data: The output data structure to be filled.
            compute_jacobian: Whether to compute the Jacobian of the objective.
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
