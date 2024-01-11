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
"""A function to compute a statistic from :class:`.ControlVariate`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from numpy import atleast_1d
from numpy import atleast_2d
from numpy import ndarray

from gemseo_umdo.formulations.functions.statistic_function import StatisticFunction

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace


class StatisticFunctionForControlVariate(StatisticFunction):
    """A function to compute a statistic from :class:`.ControlVariate`."""

    __FUNC_TEMPLATE: Final[str] = "#{}"
    """The template for the function history name from the linearization problem."""

    __GRAD_TEMPLATE: Final[str] = "##{}"
    """The template for the gradient history name from the linearization problem."""

    @property
    def _statistic_estimator_parameters(self) -> tuple[ParameterSpace]:
        return (self._formulation.uncertain_space,)

    def _compute_output_data(self, output_data: dict[str, ndarray]) -> None:
        formulation = self._formulation
        problem = formulation.mdo_formulation.opt_problem
        linearization_problem = formulation.linearization_problem
        database = problem.database
        linearization_database = linearization_problem.database
        formulation.compute_samples(problem)
        formulation.evaluate_with_mean()
        for output_name in database.get_function_names():
            output_data[output_name] = database.get_function_history(output_name)
            output_data[self.__FUNC_TEMPLATE.format(output_name)] = atleast_1d(
                linearization_database.get_function_history(output_name)
            )
            output_data[self.__GRAD_TEMPLATE.format(output_name)] = atleast_2d(
                linearization_database.get_gradient_history(output_name)
            )
        problem.reset()
        linearization_problem.reset()

    def _compute_statistic_estimation(self, output_data: dict[str, ndarray]) -> ndarray:
        output_name = self._function_name
        return self._estimate_statistic(
            output_data[output_name],
            self._formulation.doe_algo.samples,
            output_data[self.__FUNC_TEMPLATE.format(output_name)],
            output_data[self.__GRAD_TEMPLATE.format(output_name)],
        )
