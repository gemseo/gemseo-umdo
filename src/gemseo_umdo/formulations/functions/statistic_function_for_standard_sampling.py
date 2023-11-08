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

    def _func(self, input_data: ndarray) -> ndarray:
        formulation = self._formulation
        problem = formulation.mdo_formulation.opt_problem
        if (
            self._function_name in formulation._processed_functions
            or not formulation._processed_functions
        ):
            formulation._processed_functions = []
            problem.reset()
            formulation.update_top_level_disciplines(input_data)
            formulation.compute_samples(problem)

        formulation._processed_functions.append(self._function_name)
        samples, _, _ = problem.database.get_history_array(
            [self._function_name], with_x_vect=False
        )
        return self._estimate_statistic(samples)
