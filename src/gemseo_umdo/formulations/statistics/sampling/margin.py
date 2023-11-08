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
"""Estimator of a margin for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations.statistics.sampling.mean import Mean
from gemseo_umdo.formulations.statistics.sampling.sampling_estimator import (
    SamplingEstimator,
)
from gemseo_umdo.formulations.statistics.sampling.standard_deviation import (
    StandardDeviation,
)

if TYPE_CHECKING:
    from numpy import ndarray


class Margin(SamplingEstimator):
    """Estimator of a margin, i.e. mean + factor * deviation."""

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

    def __call__(self, value: ndarray) -> ndarray:  # noqa: D102
        return self.__mean(value) + self.__factor * self.__standard_deviation(value)
