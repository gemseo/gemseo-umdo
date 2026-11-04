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
    from collections.abc import Mapping

    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.util.typing import RealArray

    from gemseo_umdo.formulations._statistics.iterative_sampling.base_sampling_estimator import (  # noqa: E501
        BaseSamplingEstimator,
    )
    from gemseo_umdo.formulations.sampling import Sampling

SamplingT = TypeVar("SamplingT", bound="Sampling")

_EstimationData = tuple["RealArray", int, int | None]
"""The estimation of a statistic or its Jacobian,
the number of samples of the output used to compute it
and, for a Jacobian, the number of samples of its gradient."""


class StatisticFunctionForIterativeSampling(
    BaseStatisticFunctionForSampling[SamplingT]
):
    """A function to compute a statistic from `Sampling`."""

    _statistic_estimator: BaseSamplingEstimator

    def __init__(
        self,
        umdo_formulation: SamplingT,
        output_name: str,
        function_type: ArrayFunction.FunctionType,
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

    def _compute_statistic_estimation(
        self, data: Mapping[str, _EstimationData]
    ) -> RealArray:
        estimation, n_samples, n_gradient_samples = data[self._statistic_name]
        self._check_n_samples(n_samples, n_gradient_samples)
        return estimation

    def _compute_statistic_jacobian_estimation(
        self, data: Mapping[str, _EstimationData]
    ) -> RealArray:
        estimation, n_samples, n_gradient_samples = data[self._statistic_jac_name]
        self._check_n_samples(n_samples, n_gradient_samples)
        return estimation

    @staticmethod
    def __get_n_samples(estimation: IterativeEstimation) -> tuple[int, int | None]:
        """Return the numbers of samples used by an estimation.

        Args:
            estimation: The estimation.

        Returns:
            The number of samples of the output
            used to estimate the statistic or its Jacobian
            and, when estimating the Jacobian,
            the number of samples of its gradient,
            as passed to `_check_n_samples`.
        """
        n_samples = estimation.n_samples
        if not estimation.return_statistic_jacobian:
            return n_samples, None

        # A sample is used only if both the output and its gradient are available.
        # When the gradient is missing at all the samples,
        # the samples of the output are counted
        # to tell this failure from that of the output.
        return n_samples or estimation.n_output_samples, n_samples

    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, _EstimationData]:
        formulation = self._umdo_formulation
        # Only the callbacks of this kind of estimation run:
        # the last estimations of the others come from a previous input point.
        estimations = [
            estimation
            for estimation in self._iterative_estimations
            if estimation.return_statistic_jacobian == estimate_jacobian
        ]
        # The estimations are reset before the sampling rather than after,
        # so that a sampling interrupted by an error
        # does not leak its samples into the estimations at the next input point.
        # The estimations of all the statistic functions are reset,
        # not only those of this function,
        # as the sampling runs the callbacks of all of them
        # and the returned data are cached for all of them at this input point:
        # resetting only those of this function would make the others
        # accumulate the samples of several input points.
        # If the sampling raises an error,
        # these data are not cached
        # and the next sampling resets these estimations again.
        for estimation in estimations:
            estimation.reset()

        formulation.compute_samples(
            formulation.mdo_formulation.problem,
            input_data,
            compute_jacobian=estimate_jacobian,
        )
        estimations_to_n_samples = {
            estimation: self.__get_n_samples(estimation) for estimation in estimations
        }
        # The statistics of the same output use the same samples.
        self._log_missing_samples(
            len(formulation.doe_algo.samples),
            {
                estimation.output_name: n_samples
                for estimation, n_samples in estimations_to_n_samples.items()
            },
        )
        return {
            estimation.output_statistic_name: (estimation.last_estimation, *n_samples)
            for estimation, n_samples in estimations_to_n_samples.items()
        }
