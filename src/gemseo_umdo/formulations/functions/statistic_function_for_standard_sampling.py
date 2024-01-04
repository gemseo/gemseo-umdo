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
"""A function to compute a statistic from :class:`.Sampling`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations.functions.statistic_function_for_sampling import (
    StatisticFunctionForSampling,
)

if TYPE_CHECKING:
    from numpy import ndarray


class StatisticFunctionForStandardSampling(StatisticFunctionForSampling):
    """A function to compute a statistic from :class:`.Sampling`."""

    def _compute_statistic_estimation(self, output_data: dict[str, ndarray]) -> ndarray:
        return self._estimate_statistic(output_data[self._function_name])

    def _compute_output_data(self, output_data: dict[str, ndarray]) -> None:
        formulation = self._formulation
        problem = formulation.mdo_formulation.opt_problem
        database = problem.database
        formulation.compute_samples(problem)
        for output_name in database.get_function_names():
            output_data[output_name] = database.get_function_history(output_name)

        problem.reset()
