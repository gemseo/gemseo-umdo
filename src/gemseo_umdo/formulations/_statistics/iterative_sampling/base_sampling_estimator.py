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
"""Base statistic iterative estimator for sampling-based U-MDO formulations."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
    BaseStatisticEstimator,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray
    from openturns import IterativeAlgorithmImplementation


class BaseSamplingEstimator(BaseStatisticEstimator):
    """Base statistic iterative estimator for a U-MDO formulation using sampling.

    This class enables to iteratively compute estimators of an increasing dataset
    without storing any data in memory.
    """

    _estimator: IterativeAlgorithmImplementation
    """The iterative estimator of the statistic."""

    _size: int
    """The size of the output of interest."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self._size = 0

    @abstractmethod
    def reset(self) -> None:
        """Reset the estimator of the statistic."""

    def _get_statistic(self) -> RealArray | None:
        """Return the statistic.

        Returns:
            The current estimation of the statistic if required;
            otherwise ``None``.
        """

    def __call__(self, value: RealArray) -> RealArray:
        """
        Args:
            value: The value to update the estimation of the statistic.
        """  # noqa: D205 D212 D415
        if self._size == 0:
            self._size = value.size
            self.reset()

        self._estimator.increment(value)
        return self._get_statistic()
