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

from gemseo_umdo.formulation._estimator.iterative_sampling.base import (
    BaseIterativeSamplingEstimator,
)
from gemseo_umdo.formulation._estimator.iterative_sampling.mean import Mean
from gemseo_umdo.formulation._estimator.iterative_sampling.standard_deviation import (
    StandardDeviation,
)
from gemseo_umdo.statistic.settings.margin import Margin_Settings
from gemseo_umdo.statistic.settings.mean import Mean_Settings
from gemseo_umdo.statistic.settings.standard_deviation import StandardDeviation_Settings

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class Margin(BaseIterativeSamplingEstimator[Margin_Settings]):
    """Iterative estimator of a margin, i.e. mean + factor * deviation."""

    __mean: Mean
    """The iterative estimator of the mean."""

    __standard_deviation: StandardDeviation
    """The iterative estimator of the standard deviation."""

    def __init__(self, settings: Margin_Settings) -> None:
        """
        Args:
            settings: The settings of the margin.
        """  # noqa: D205 D212 D415
        super().__init__(settings)
        self.__mean = Mean(Mean_Settings())
        self.__standard_deviation = StandardDeviation(StandardDeviation_Settings())

    @property
    def min_n_samples(self) -> int:  # noqa: D102
        return self.__standard_deviation.min_n_samples

    def estimate_statistic(self, value: RealArray) -> RealArray:
        self.shape = value.shape
        return self.__mean.estimate_statistic(
            value
        ) + self._settings.factor * self.__standard_deviation.estimate_statistic(value)

    def compute_jacobian(self, value: RealArray, jac_value: RealArray) -> RealArray:
        self.jac_shape = jac_value.shape
        dmean_dx = self.__mean.compute_jacobian(value, jac_value)
        dstd_dx = self.__standard_deviation.compute_jacobian(value, jac_value)
        return dmean_dx + self._settings.factor * dstd_dx

    def reset(self) -> None:  # noqa: D102
        self.__mean.reset()
        self.__standard_deviation.reset()
