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
"""Base statistic estimator for U-MDO formulation based on control variates."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Final

from numpy import finfo

from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
    BaseStatisticEstimator,
)

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.typing import RealArray


class BaseControlVariateEstimator(BaseStatisticEstimator):
    """Base statistic estimator for a U-MDO formulation using control variates."""

    __EPSILON: Final[float] = finfo(float).eps
    """A number to avoid division by zero when normalizing the covariance."""

    def __init__(self, uncertain_space: ParameterSpace) -> None:
        """
        Args:
            uncertain_space: The uncertain space.
        """  # noqa: D205 D212 D415
        self._uncertain_space = uncertain_space

    @abstractmethod
    def estimate_statistic(
        self,
        evaluations: RealArray,
        mean: RealArray,
        variance: RealArray,
        some_predictions: RealArray,
        many_predictions: RealArray,
    ) -> RealArray:
        """
        Args:
            evaluations: The output evaluations arranged in rows.
            mean: The mean of the regressor's output vector.
            variance: The variance of the regressor's output vector.
            some_predictions: Some output evaluations of the regressor arranged in rows.
            many_predictions: Many output evaluations of the regressor arranged in rows.
        """  # noqa: D205 D212 D415

    @classmethod
    def _compute_opposite_scaled_covariance(
        cls, data: RealArray, cv_data: RealArray
    ) -> RealArray:
        """Compute the opposite scaled covariance between data and control variate data.

        Args:
            data: The high-fidelity data.
            cv_data: The control variate data.

        Returns:
            The opposite of the scaled covariance between data and control variate data.
        """
        centered_cv_data = cv_data - cv_data.mean(axis=0)
        den = (centered_cv_data**2).sum(axis=0)
        den[den < cls.__EPSILON] = cls.__EPSILON
        return -((data - data.mean(axis=0)) * centered_cv_data).sum(axis=0) / den
