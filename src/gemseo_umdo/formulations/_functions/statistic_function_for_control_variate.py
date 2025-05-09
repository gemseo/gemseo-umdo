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
"""A function to compute a statistic from `ControlVariate`.

See also [ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate].
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import TypeVar

from numpy import atleast_1d
from numpy import atleast_2d
from numpy import newaxis

from gemseo_umdo.formulations._functions.base_statistic_function import (
    BaseStatisticFunction,
)

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.control_variate import ControlVariate

ControlVariateT = TypeVar("ControlVariateT", bound="ControlVariate")


class StatisticFunctionForControlVariate(BaseStatisticFunction[ControlVariateT]):
    """A function to compute a statistic from `ControlVariate`."""

    __FUNC_TEMPLATE: Final[str] = "#{}"
    """The template for the function history name from the linearization problem."""

    __JAC_TEMPLATE: Final[str] = "##{}"
    """The template for the Jacobian history name from the linearization problem."""

    __mc_problem: OptimizationProblem
    """The evaluation problem defined over the uncertain space for sampling."""

    __mean_input_value: RealArray
    """The mean value of the uncertain vector."""

    __problem: OptimizationProblem
    """The evaluation problem defined over the uncertain space for Taylor."""

    def __init__(
        self,
        umdo_formulation: ControlVariateT,
        output_name: str,
        function_type: MDOFunction.FunctionType,
        statistic_operator_name: str,
        **statistic_options: Any,
    ) -> None:
        super().__init__(
            umdo_formulation,
            output_name,
            function_type,
            statistic_operator_name,
            **statistic_options,
        )
        self.__mc_problem = self._umdo_formulation.mdo_formulation.optimization_problem
        formulation = self._umdo_formulation
        self.__mean_input_value = formulation.uncertain_space.distribution.mean
        self.__problem = formulation.auxiliary_mdo_formulation.optimization_problem

    @property
    def _statistic_estimator_parameters(self) -> tuple[ParameterSpace]:
        return (self._umdo_formulation.uncertain_space,)

    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, Any]:
        self._umdo_formulation.compute_samples(self.__mc_problem)
        get_history = self.__mc_problem.database.get_function_history
        mean_input_value = self.__mean_input_value
        func_template = self.__FUNC_TEMPLATE
        grad_template = self.__JAC_TEMPLATE
        data = {}
        for function in self.__problem.functions:
            data[output_name] = get_history(output_name := function.name)
            data[func_template.format(output_name)] = atleast_1d(
                function.evaluate(mean_input_value)
            )
            data[grad_template.format(output_name)] = atleast_2d(
                function.jac(mean_input_value)
            )

        return data

    def _compute_statistic_estimation(self, data: dict[str, RealArray]) -> RealArray:
        output_name = self._output_name
        samples = data[output_name]
        if samples.ndim == 1:
            samples = samples[:, newaxis]

        return self._statistic_estimator.estimate_statistic(
            samples,
            self._umdo_formulation.doe_algo.samples,
            data[self.__FUNC_TEMPLATE.format(output_name)],
            data[self.__JAC_TEMPLATE.format(output_name)],
        )
