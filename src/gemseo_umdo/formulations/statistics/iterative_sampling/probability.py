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
# Copyright 2021 IRT Saint Exupéry,7 https://www.irt-saintexupery.com
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

from numpy import array
from numpy import ndarray
from openturns import IterativeThresholdExceedance

from gemseo_umdo.formulations.statistics.iterative_sampling.sampling_estimator import (
    SamplingEstimator,
)


class Probability(SamplingEstimator):
    """Iterative estimator of a probability.

    This class iteratively computes a probability on an increasing dataset without
    storing any data in memory.
    """

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

    def _get_statistic(self) -> ndarray:
        result = array(
            self._estimator.getThresholdExceedance()
            / self._estimator.getIterationNumber()
        )
        return result if self.__greater else 1 - result

    def reset(self) -> None:  # noqa: D102
        self._estimator = IterativeThresholdExceedance(self._size, self.__threshold)
