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
"""A function to compute a statistic from `Sampling`.

See also [Sampling][gemseo_umdo.formulations.sampling.Sampling].
"""

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
    """A function to compute a statistic from `Sampling`."""

    def _update_sampling_problem(
        self, sampling_problem: OptimizationProblem, function: MDOFunction
    ) -> None:
        self._formulation._estimators.append((
            self._observable_name,
            self._estimate_statistic,
        ))
        sampling_problem.add_observable(
            IterativeEstimation(
                self._observable_name, function, self._estimate_statistic
            )
        )

    def _compute_statistic_estimation(self, output_data: dict[str, ndarray]) -> ndarray:
        return output_data[self._observable_name]

    def _compute_output_data(self, output_data: dict[str, ndarray]) -> None:
        formulation = self._formulation
        problem = formulation.mdo_formulation.opt_problem
        formulation.compute_samples(problem)
        for estimator_name, estimate_statistic in self._formulation._estimators:
            estimator_function = problem.get_observable(estimator_name)
            output_data[estimator_name] = estimator_function.last_eval
            estimate_statistic.reset()

        problem.reset(preprocessing=False)
