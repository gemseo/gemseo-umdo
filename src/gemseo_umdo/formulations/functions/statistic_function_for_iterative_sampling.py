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
"""A function to compute a statistic from :class:`.Sampling` iteratively."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations.functions.iterative_estimation import IterativeEstimation
from gemseo_umdo.formulations.functions.statistic_function_for_sampling import (
    StatisticFunctionForSampling,
)

if TYPE_CHECKING:
    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from numpy import ndarray


class StatisticFunctionForIterativeSampling(StatisticFunctionForSampling):
    """A function to compute a statistic from :class:`.Sampling` iteratively."""

    def _update_sampling_problem(
        self, sampling_problem: OptimizationProblem, function: MDOFunction
    ) -> None:
        self._formulation._estimators.append(self._estimate_statistic)
        sampling_problem.add_observable(
            IterativeEstimation(
                self._observable_name, function, self._estimate_statistic
            )
        )

    def _func(self, input_data: ndarray) -> ndarray:
        formulation = self._formulation
        problem = formulation.mdo_formulation.opt_problem
        if (
            self._function_name in formulation._processed_functions
            or not formulation._processed_functions
        ):
            formulation._processed_functions = []
            for estimator in self._formulation._estimators:
                estimator.reset()
            problem.reset()
            formulation.update_top_level_disciplines(input_data)
            formulation.compute_samples(problem)

        formulation._processed_functions.append(self._function_name)
        return problem.get_observable(self._observable_name).last_eval
