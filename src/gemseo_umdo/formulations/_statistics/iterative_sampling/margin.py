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
"""Iterative estimator of a margin for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.iterative_sampling.base_sampling_estimator import (  # noqa: E501
    BaseSamplingEstimator,
)
from gemseo_umdo.formulations._statistics.iterative_sampling.mean import Mean
from gemseo_umdo.formulations._statistics.iterative_sampling.standard_deviation import (
    StandardDeviation,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Margin(BaseSamplingEstimator):
    """Iterative estimator of a margin, i.e. mean + factor * deviation."""

    __factor: float
    """The factor related to the standard deviation."""

    __mean: Mean
    """The iterative estimator of the mean."""

    __standard_deviation: StandardDeviation
    """The iterative estimator of the standard deviation."""

    def __init__(self, factor: float = 2.0) -> None:
        """
        Args:
            factor: The factor related to the standard deviation.
        """  # noqa: D205 D212 D415
        super().__init__()
        self.__mean = Mean()
        self.__standard_deviation = StandardDeviation()
        self.__factor = factor

    def estimate_statistic(self, value: RealArray) -> RealArray:
        self.shape = value.shape
        return self.__mean.estimate_statistic(
            value
        ) + self.__factor * self.__standard_deviation.estimate_statistic(value)

    def compute_jacobian(self, value: RealArray, jac_value: RealArray) -> RealArray:
        self.jac_shape = jac_value.shape
        dmean_dx = self.__mean.compute_jacobian(value, jac_value)
        dstd_dx = self.__standard_deviation.compute_jacobian(value, jac_value)
        return dmean_dx + self.__factor * dstd_dx

    def reset(self, size: int) -> None:  # noqa: D102
        super().reset(size)
        self.__mean.reset(size)
        self.__standard_deviation.reset(size)
