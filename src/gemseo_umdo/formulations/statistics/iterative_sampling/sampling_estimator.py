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
"""Statistic iterative estimator for a U-MDO formulation using sampling."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo_umdo.formulations.statistics.iterative_sampling.base_sampling_estimator import (  # noqa: E501
    BaseSamplingEstimator,
)

if TYPE_CHECKING:
    from numpy import ndarray


class SamplingEstimator(BaseSamplingEstimator):
    """Statistic iterative estimator for a U-MDO formulation using sampling.

    Derive this class to implement estimator of simple statistics,
    e.g. mean, standard deviation, variance and probability.

    Derive from :class:`.BaseSamplingEstimator` for combinations of statistics,
    e.g. margin.
    """

    @abstractmethod
    def _get_statistic(self) -> ndarray:
        """Return the statistic.

        Returns:
            The current estimation of the statistic.
        """

    def __call__(self, value: ndarray) -> ndarray:
        """
        Args:
            value: The value to update the estimation of the statistic.
        """  # noqa: D205 D212 D415
        if self._size == 0:
            self._size = value.size
            self.reset()

        self._estimator.increment(value)
        return self._get_statistic()
