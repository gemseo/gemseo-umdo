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

from gemseo_umdo.formulation._statistic.base_margin import BaseMargin
from gemseo_umdo.formulation._statistic.sampling.base_sampling_estimator import (
    BaseSamplingEstimator,
)
from gemseo_umdo.formulation._statistic.sampling.mean import Mean
from gemseo_umdo.formulation._statistic.sampling.standard_deviation import (
    StandardDeviation,
)

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class Margin(BaseMargin, BaseSamplingEstimator):
    """Estimator of a margin, i.e. mean + factor * deviation."""

    __factor: float
    """The factor related to the standard deviation."""

    __mean: Mean
    """The iterative estimator of the mean."""

    __standard_deviation: StandardDeviation
    """The iterative estimator of the standard deviation."""

    def __init__(self, factor: float) -> None:
        """
        Args:
            factor: The factor related to the standard deviation.
        """  # noqa: D205 D212 D415
        super().__init__()
        self.__mean = Mean()
        self.__standard_deviation = StandardDeviation()
        self.__factor = factor

    @property
    def min_n_samples(self) -> int:  # noqa: D102
        return self.__standard_deviation.min_n_samples

    def estimate_statistic(self, samples: RealArray) -> RealArray:
        mean = self.__mean.estimate_statistic(samples)
        standard_deviation = self.__standard_deviation.estimate_statistic(samples)
        return mean + self.__factor * standard_deviation

    def compute_jacobian(self, samples: RealArray, jac_samples: RealArray) -> RealArray:
        mean_jac = self.__mean.compute_jacobian(samples, jac_samples)
        standard_deviation_jac = self.__standard_deviation.compute_jacobian(
            samples, jac_samples
        )
        return mean_jac + self.__factor * standard_deviation_jac
