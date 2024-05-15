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

from numpy import cov
from numpy import finfo
from numpy import var

from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
    BaseStatisticEstimator,
)

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.typing import RealArray
    from numpy.typing import NDArray


class ControlVariateEstimator(BaseStatisticEstimator):
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
    def __call__(
        self, samples: RealArray, u_samples: RealArray, mean: RealArray, jac: RealArray
    ) -> RealArray:
        """
        Args:
            samples: The output evaluations arranged in rows.
            u_samples: The input evaluations arranged in rows.
            mean: The output value at the mean input one.
            jac: The Jacobian value at the mean input one.
        """  # noqa: D205 D212 D415

    def _compute_lf_and_hf_samples(
        self, samples: RealArray, u_samples: RealArray, mean: RealArray, jac: RealArray
    ) -> tuple[RealArray, RealArray]:
        """Compute the low- and high-fidelity samples.

        Args:
            samples: The output evaluations arranged in rows.
            u_samples: The input evaluations arranged in rows.
            mean: The output value at the mean input one.
            jac: The Jacobian value at the mean input one.

        Returns:
            The low- and high-fidelity samples.
        """
        return (mean + (u_samples - self._u_mean) @ jac.T).ravel(), samples.ravel()

    @classmethod
    def _compute_opposite_scaled_covariance(
        cls, h_f: RealArray, l_f: RealArray
    ) -> float:
        """Compute the opposite scaled covariance between high- and low-fidelity data.

        Args:
            h_f: The high-fidelity data.
            l_f: The low-fidelity data.

        Returns:
            The opposite of the scaled covariance
            between high- and low-fidelity data.
        """
        return -cov(h_f, l_f)[0, 1] / max(var(l_f), cls.__EPSILON)
