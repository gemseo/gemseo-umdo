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

from gemseo.algos.hashable_ndarray import HashableNdarray
from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
        BaseStatisticEstimator,
    )
    from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation


class BaseStatisticFunction(MDOFunction):
    """A function to compute a statistic from a `BaseUMDOFormulation`."""

    _estimate_statistic: BaseStatisticEstimator
    """A callable to estimate the statistic."""

    _formulation: BaseUMDOFormulation
    """The U-MDO formulation to which the
    [BaseStatisticFunction][gemseo_umdo.formulations.f
    unctions.base_statistic_function.BaseStatisticFunction] is attached."""

    _function_name: str
    """The name of the function."""

    def __init__(
        self,
        formulation: BaseUMDOFormulation,
        func: MDOFunction,
        function_type: MDOFunction.FunctionType,
        name: str,
        sampling_problem: OptimizationProblem,
        **statistic_options: Any,
    ) -> None:
        """
        Args:
            formulation: The U-MDO formulation
                to which the
                [BaseStatisticFunction][gemseo_umdo.formulations._functions.base_statistic_function.BaseStatisticFunction]
                is attached.
            func: The function for which we want to estimate an output statistic.
            function_type: The type of function.
            name: The name of the statistic.
            sampling_problem: The problem
                evaluating the function over the uncertain space.
            **statistic_options: The options of the statistic.
        """  # noqa: D205 D212 D415
        self._function_name = func.name
        self._formulation = formulation
        self._estimate_statistic = formulation._statistic_factory.create(
            name, *self._statistic_estimator_parameters, **statistic_options
        )
        self._update_sampling_problem(sampling_problem, func)
        super().__init__(self._func, name=func.name, f_type=function_type)

    @property
    def _statistic_estimator_parameters(self) -> tuple[Any]:
        """The parameters of the estimator of the statistic."""
        return ()

    def _update_sampling_problem(
        self, sampling_problem: OptimizationProblem, function: MDOFunction
    ) -> None:
        """Update the problem evaluating the function over the uncertain space.

        Args:
            sampling_problem: The problem
                evaluating the function over the uncertain space.
            function: The function to update the problem.
        """

    def _func(self, input_data: RealArray) -> RealArray:
        """A function estimating the statistic at a given input point.

        Args:
            input_data: The input point at which to estimate the statistic.

        Returns:
            The estimation of the statistic at the given input point.
        """
        formulation = self._formulation
        hashable_input_data = HashableNdarray(input_data)
        output_data = formulation.input_data_to_output_data.get(hashable_input_data)
        if output_data is None:
            formulation.update_top_level_disciplines(input_data)
            output_data = formulation.input_data_to_output_data[
                hashable_input_data
            ] = {}
            self._compute_output_data(input_data, output_data)

        return self._compute_statistic_estimation(output_data)

    @abstractmethod
    def _compute_statistic_estimation(self, data: dict[str, Any]) -> RealArray:
        """Estimate the statistic.

        Args:
            data: The data from which to estimate the statistic.

        Returns:
            The estimation of the statistic.
        """

    @abstractmethod
    def _compute_output_data(
        self, input_data: RealArray, output_data: dict[str, Any]
    ) -> None:
        """Compute the output data.

        Args:
            input_data: The input point at which to estimate the statistic.
            output_data: The output data structure to be filled.
        """
