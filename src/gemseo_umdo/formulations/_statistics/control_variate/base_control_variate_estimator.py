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
    from numpy.typing import NDArray


class BaseControlVariateEstimator(BaseStatisticEstimator):
    """Base statistic estimator for a U-MDO formulation using control variates."""

    __EPSILON: Final[float] = finfo(float).eps
    """A number to avoid division by zero when normalizing the covariance."""

    _u_mean: NDArray[float]
    """The input mean vector."""

    _u_standard_deviation: NDArray[float]
    """The input standard deviation vector."""

    _uncertain_space: ParameterSpace
    """The uncertain space."""

    def __init__(self, uncertain_space: ParameterSpace) -> None:
        """
        Args:
            uncertain_space: The uncertain space.
        """  # noqa: D205 D212 D415
        self._u_mean = uncertain_space.distribution.mean
        self._u_standard_deviation = uncertain_space.distribution.standard_deviation
        self._uncertain_space = uncertain_space

    @abstractmethod
    def estimate_statistic(
        self,
        samples: RealArray,
        u_samples: RealArray,
        mean: RealArray,
        jac: RealArray,
    ) -> RealArray:
        """
        Args:
            samples: The output evaluations arranged in rows.
            u_samples: The input evaluations arranged in rows.
            mean: The output value at the mean input one.
            jac: The Jacobian value at the mean input one.
        """  # noqa: D205 D212 D415

    def _compute_control_variate_samples(
        self, u_samples: RealArray, mean: RealArray, jac: RealArray
    ) -> RealArray:
        """Compute the samples of the control variates.

        Args:
            u_samples: The input evaluations arranged in rows.
            mean: The output value at the mean input one.
            jac: The Jacobian value at the mean input one.

        Returns:
            The samples of the control variates.
        """
        return mean + (u_samples - self._u_mean) @ jac.T

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
