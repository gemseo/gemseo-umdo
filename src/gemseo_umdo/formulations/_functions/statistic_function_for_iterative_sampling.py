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
from typing import TypeVar

from gemseo_umdo.formulations._functions.base_statistic_function_for_sampling import (
    BaseStatisticFunctionForSampling,
)
from gemseo_umdo.formulations._functions.iterative_estimation import IterativeEstimation

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations._statistics.iterative_sampling.base_sampling_estimator import (  # noqa: E501
        BaseSamplingEstimator,
    )
    from gemseo_umdo.formulations.sampling import Sampling

SamplingT = TypeVar("SamplingT", bound="Sampling")


class StatisticFunctionForIterativeSampling(
    BaseStatisticFunctionForSampling[SamplingT]
):
    """A function to compute a statistic from `Sampling`."""

    _estimate_statistic: BaseSamplingEstimator

    def _update_sampling_problem(
        self, sampling_problem: OptimizationProblem, function: MDOFunction
    ) -> None:
        self._formulation._estimators.append((
            self._observable_name,
            self._estimate_statistic,
        ))
        self._formulation.callbacks.append(
            IterativeEstimation(function.name, self._estimate_statistic)
        )

    def _compute_statistic_estimation(
        self, output_data: dict[str, RealArray]
    ) -> RealArray:
        return output_data[self._observable_name]

    def _compute_output_data(
        self, input_data: RealArray, output_data: dict[str, RealArray]
    ) -> None:
        formulation = self._formulation
        problem = formulation.mdo_formulation.optimization_problem
        formulation.compute_samples(problem, input_data)
        for (estimator_name, estimate_statistic), iterative_estimation in zip(
            formulation._estimators, formulation.callbacks
        ):
            output_data[estimator_name] = iterative_estimation.last_estimation
            estimate_statistic.reset()

        problem.reset(preprocessing=False)
