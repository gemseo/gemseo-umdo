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
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.taylor_polynomial import TaylorPolynomial

TaylorPolynomialT = TypeVar("TaylorPolynomialT", bound="TaylorPolynomial")


class StatisticFunctionForTaylorPolynomial(BaseStatisticFunction[TaylorPolynomialT]):
    """A function to compute a statistic from `TaylorPolynomial`."""

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
        reset_problems: bool = False,
    ) -> None:
        formulation = self._umdo_formulation
        problem = formulation.auxiliary_mdo_formulation.optimization_problem
        database = problem.database
        formulation.evaluate_with_mean(problem, True)
        for function in problem.functions:
            name = function.name
            output_data[name] = atleast_1d(function.last_eval)
            output_data[database.get_gradient_name(name)] = atleast_2d(
                database.get_gradient_history(self._function_name)[0]
            )

        if formulation.second_order:
            formulation.evaluate_with_mean(formulation.hessian_fd_problem, False)
            hessian_database = formulation.hessian_fd_problem.database
            for function in problem.functions:
                hess_name = database.get_gradient_name(
                    database.get_gradient_name(function.name)
                )
                hess_value = hessian_database.get_function_history(hess_name)[0]
                if hess_value.ndim == 1:
                    hess_value = hess_value[newaxis, newaxis, ...]

                if hess_value.ndim == 2:
                    hess_value = hess_value[newaxis, ...]

                output_data[hess_name] = hess_value

    @property
    def _other_evaluation_problems(self) -> tuple[EvaluationProblem, ...]:
        return (
            (self._umdo_formulation.hessian_fd_problem,)
            if self._umdo_formulation.second_order
            else ()
        )
