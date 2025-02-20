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

    __GRAD_TEMPLATE: Final[str] = "##{}"
    """The template for the gradient history name from the linearization problem."""

    __mc_problem: OptimizationProblem
    """The evaluation problem defined over the uncertain space for sampling."""

    __mean_input_value: RealArray
    """The mean value of the uncertain vector."""

    __problem: OptimizationProblem
    """The evaluation problem defined over the uncertain space for Taylor."""

    def __init__(
        self,
        umdo_formulation: ControlVariateT,
        function: MDOFunction,
        function_type: MDOFunction.FunctionType,
        name: str,
        **statistic_options: Any,
    ) -> None:
        super().__init__(
            umdo_formulation, function, function_type, name, **statistic_options
        )
        self.__mc_problem = self._umdo_formulation.mdo_formulation.optimization_problem
        formulation = self._umdo_formulation
        self.__mean_input_value = formulation.uncertain_space.distribution.mean
        self.__problem = formulation.auxiliary_mdo_formulation.optimization_problem

    @property
    def _statistic_estimator_parameters(self) -> tuple[ParameterSpace]:
        return (self._umdo_formulation.uncertain_space,)

    def _compute_output_data(
        self,
        input_data: RealArray,
        output_data: dict[str, RealArray],
        compute_jacobian: bool = False,
    ) -> None:
        self._umdo_formulation.compute_samples(self.__mc_problem)
        database = self.__mc_problem.database
        for function in self.__problem.functions:
            output_name = function.name
            output_data[output_name] = database.get_function_history(output_name)
            output_data[self.__FUNC_TEMPLATE.format(output_name)] = atleast_1d(
                function.evaluate(self.__mean_input_value)
            )
            output_data[self.__GRAD_TEMPLATE.format(output_name)] = atleast_2d(
                function.jac(self.__mean_input_value)
            )

    def _compute_statistic_estimation(
        self, output_data: dict[str, RealArray]
    ) -> RealArray:
        output_name = self._function_name
        samples = output_data[output_name]
        if samples.ndim == 1:
            samples = samples[:, newaxis]
        return self._statistic_estimator.estimate_statistic(
            samples,
            self._umdo_formulation.doe_algo.samples,
            output_data[self.__FUNC_TEMPLATE.format(output_name)],
            output_data[self.__GRAD_TEMPLATE.format(output_name)],
        )
