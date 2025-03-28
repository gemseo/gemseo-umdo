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
"""A function to compute a statistic from `TaylorPolynomial`.

See also [TaylorPolynomial][gemseo_umdo.formulations.taylor_polynomial.TaylorPolynomial].
"""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import TypeVar

from gemseo.algos.database import Database
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import newaxis

from gemseo_umdo.formulations._functions.base_statistic_function import (
    BaseStatisticFunction,
)

if TYPE_CHECKING:
    from gemseo.algos.evaluation_problem import EvaluationProblem
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.taylor_polynomial import TaylorPolynomial

TaylorPolynomialT = TypeVar("TaylorPolynomialT", bound="TaylorPolynomial")


class StatisticFunctionForTaylorPolynomial(BaseStatisticFunction[TaylorPolynomialT]):
    """A function to compute a statistic from `TaylorPolynomial`."""

    __get_gradient_name: Callable[[str], str]
    """The function returning the name of the Jacobian function from the output name."""

    __mean_input_value: RealArray
    """The mean value of the uncertain vector."""

    __problem: EvaluationProblem
    """The evaluation problem defined over the uncertain space for Taylor."""

    def __init__(
        self,
        umdo_formulation: TaylorPolynomialT,
        function: MDOFunction,
        function_type: MDOFunction.FunctionType,
        name: str,
        **statistic_options: Any,
    ) -> None:
        super().__init__(
            umdo_formulation, function, function_type, name, **statistic_options
        )
        formulation = self._umdo_formulation
        self.__problem = formulation.auxiliary_mdo_formulation.optimization_problem
        self.__get_gradient_name = self.__problem.database.get_gradient_name
        self.__mean_input_value = formulation.uncertain_space.distribution.mean

    @property
    def _statistic_estimator_parameters(self) -> tuple[ParameterSpace]:
        return (self._umdo_formulation.uncertain_space,)

    def _compute_statistic_estimation(
        self, output_data: dict[str, RealArray]
    ) -> RealArray:
        function_name = self._function_name
        gradient_name = Database.get_gradient_name(function_name)
        return self._statistic_estimator.estimate_statistic(
            output_data[function_name],
            output_data[gradient_name],
            output_data.get(Database.get_gradient_name(gradient_name)),
        )

    def _compute_output_data(
        self,
        input_data: RealArray,
        output_data: dict[str, RealArray],
        compute_jacobian: bool = False,
    ) -> None:
        for function in self.__problem.functions:
            name = function.name
            output_data[name] = atleast_1d(function.evaluate(self.__mean_input_value))
            output_data[self.__get_gradient_name(name)] = atleast_2d(
                function.jac(self.__mean_input_value)
            )

        if self._umdo_formulation.second_order:
            hessian_fd_problem = self._umdo_formulation.hessian_fd_problem
            for function, hessian_function in zip(
                self.__problem.functions, hessian_fd_problem.functions
            ):
                hess_value = hessian_function.evaluate(self.__mean_input_value)
                if hess_value.ndim == 2:
                    hess_value = hess_value[newaxis, ...]

                hess_name = self.__get_gradient_name(
                    self.__get_gradient_name(function.name)
                )
                output_data[hess_name] = hess_value

    @property
    def _other_evaluation_problems(self) -> tuple[EvaluationProblem, ...]:
        return (
            (self._umdo_formulation.hessian_fd_problem,)
            if self._umdo_formulation.second_order
            else ()
        )
