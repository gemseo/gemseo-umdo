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
"""Iterative estimator of a probability for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array

from gemseo_umdo._utils.compatibility.openturns import IterativeThresholdExceedance
from gemseo_umdo.formulations._statistics.iterative_sampling.base_sampling_estimator import (  # noqa: E501
    BaseSamplingEstimator,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Probability(BaseSamplingEstimator):
    """Iterative estimator of a probability."""

    _estimator: IterativeThresholdExceedance | None

    __greater: bool
    """Whether the probability is linked to exceeding the threshold."""

    __threshold: float
    """The threshold against which the probability is estimated."""

    def __init__(
        self,
        threshold: float = 0.0,
        greater: bool = True,
    ) -> None:
        """
        Args:
            threshold: The threshold against which the probability is estimated.
            greater: Whether to compute the probability of exceeding the threshold.
        """  # noqa: D205 D212 D415
        super().__init__()
        self.__threshold = threshold
        self.__greater = greater

    def _get_estimation(self) -> RealArray:
        result = array(
            self._estimator.getThresholdExceedance()
            / self._estimator.getIterationNumber()
        )
        return result if self.__greater else 1 - result

    def reset(self, size: int) -> None:  # noqa: D102
        super().reset(size)
        self._estimator = IterativeThresholdExceedance(size, self.__threshold)
