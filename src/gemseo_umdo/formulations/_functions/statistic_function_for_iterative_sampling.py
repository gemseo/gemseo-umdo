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
from typing import Any
from typing import TypeVar

from gemseo_umdo.formulations._functions.base_statistic_function_for_sampling import (
    BaseStatisticFunctionForSampling,
)
from gemseo_umdo.formulations._functions.iterative_estimation import IterativeEstimation

if TYPE_CHECKING:
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
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

    _statistic_estimator: BaseSamplingEstimator

    def __init__(
        self,
        umdo_formulation: SamplingT,
        output_name: str,
        function_type: MDOFunction.FunctionType,
        statistic_operator_name: str,
        **statistic_options: Any,
    ) -> None:
        super().__init__(
            umdo_formulation,
            output_name,
            function_type,
            statistic_operator_name,
            **statistic_options,
        )
        self._umdo_formulation.callbacks.append(
            IterativeEstimation(
                output_name, self._statistic_name, self._statistic_estimator
            )
        )
        self._umdo_formulation.jacobian_callbacks.append(
            IterativeEstimation(
                output_name,
                self._statistic_jac_name,
                self._statistic_estimator,
                return_statistic_jacobian=True,
            )
        )

    @property
    def _iterative_estimations(self) -> tuple[IterativeEstimation, ...]:
        """The iterative estimation objects from OpenTURNS."""
        return tuple(
            callback
            for callbacks in zip(
                self._umdo_formulation.callbacks,
                self._umdo_formulation.jacobian_callbacks,
                strict=False,
            )
            for callback in callbacks
            if isinstance(callback, IterativeEstimation)
        )

    def _compute_statistic_estimation(self, data: dict[str, RealArray]) -> RealArray:
        return data[self._statistic_name]

    def _compute_statistic_jacobian_estimation(
        self, data: dict[str, RealArray]
    ) -> RealArray:
        return data[self._statistic_jac_name]

    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, Any]:
        formulation = self._umdo_formulation
        formulation.compute_samples(
            formulation.mdo_formulation.optimization_problem,
            input_data,
            compute_jacobian=estimate_jacobian,
        )
        data = {}
        for estimation in self._iterative_estimations:
            if estimation.return_statistic_jacobian and not estimate_jacobian:
                continue

            data[estimation.output_statistic_name] = estimation.last_estimation
            estimation.statistic_estimator.reset()

        return data
