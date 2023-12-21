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
"""Iterative estimator of a moment for sampling-based U-MDO formulations."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from numpy import array
from numpy import ndarray
from openturns import IterativeMoments
from openturns import Point

from gemseo_umdo.formulations.statistics.iterative_sampling.sampling_estimator import (
    SamplingEstimator,
)


class CentralMoment(SamplingEstimator):
    """Iterative estimator of a central moment, e.g. expectation or variance.

    This class iteratively computes a central moment of an increasing dataset without
    storing any data in memory.
    """

    _ORDER: ClassVar[int]
    """The order of the central moment."""

    def _get_statistic(self) -> ndarray:
        return array(self._get_central_moment())

    @abstractmethod
    def _get_central_moment(self) -> Point:
        """Return the current value of the central moment estimated iteratively."""

    def reset(self) -> None:  # noqa: D102
        self._estimator = IterativeMoments(self._ORDER, self._size)
