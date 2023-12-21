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
"""A function to compute a statistic from :class:`.TaylorPolynomial`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import atleast_1d
from numpy import atleast_2d
from numpy import ndarray
from numpy import newaxis

from gemseo_umdo.formulations.functions.statistic_function import StatisticFunction

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace


class StatisticFunctionForTaylorPolynomial(StatisticFunction):
    """A function to compute a statistic from :class:`.TaylorPolynomial`."""

    @property
    def _statistic_estimator_parameters(self) -> tuple[ParameterSpace]:
        return (self._formulation.uncertain_space,)

    def _func(self, input_data: ndarray) -> ndarray:
        formulation = self._formulation
        problem = formulation.mdo_formulation.opt_problem
        if self._function_name in formulation._processed_functions:
            formulation._processed_functions = []
            problem.reset()
            if formulation.hessian_fd_problem is not None:
                formulation.hessian_fd_problem.reset()

        database = problem.database
        if not database:
            formulation.update_top_level_disciplines(input_data)
            formulation.evaluate_with_mean(problem, True)
            if formulation.hessian_fd_problem is not None:
                formulation.evaluate_with_mean(formulation.hessian_fd_problem, False)

        func_value = atleast_1d(database.get_function_history(self._function_name)[0])
        jac_value = atleast_2d(database.get_gradient_history(self._function_name)[0])
        hess_value = None
        if formulation.second_order:
            hessian_database = formulation.hessian_fd_problem.database
            hess_name = f"{database.GRAD_TAG}{database.GRAD_TAG}{self._function_name}"
            hess_value = hessian_database.get_function_history(hess_name)[0]
            if hess_value.ndim == 1:
                hess_value = hess_value[newaxis, newaxis, ...]

            if hess_value.ndim == 2:
                hess_value = hess_value[newaxis, ...]

        formulation._processed_functions.append(self._function_name)
        return self._estimate_statistic(func_value, jac_value, hess_value)
