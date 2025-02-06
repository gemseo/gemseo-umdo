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

from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
    BaseStatisticEstimator,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray
    from openturns import IterativeAlgorithmImplementation


class BaseSamplingEstimator(BaseStatisticEstimator):
    """Base statistic iterative estimator for a U-MDO formulation using sampling."""

    _estimator: IterativeAlgorithmImplementation | None
    """The iterative estimator of the statistic.

    `None` before the first call to `estimate_statistic`.
    """

    _jac_estimator: IterativeAlgorithmImplementation | None
    """The iterative estimator of the Jacobian of the statistic.

    `None` before the first call to `compute_jacobian`.
    """

    shape: tuple[int] | None
    """The shape of the output of interest, if known."""

    jac_shape: tuple[int, int] | None
    """The shape of the Jacobian, if known."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.shape = None
        self.jac_shape = None
        self._estimator = None
        self._jac_estimator = None

    def reset(self, size: int) -> None:
        """Reset the estimator of the statistic."""

    def _get_estimation(self) -> RealArray | None:
        """Return the current estimation of the statistic.

        Returns:
            The current estimation of the statistic if required;
            otherwise ``None``.
        """

    def _get_estimation_jacobian(self) -> RealArray | None:
        """Return the Jacobian of the current estimation of the statistic.

        Returns:
            The Jacobian of the current estimation of the statistic if required;
            otherwise ``None``.
        """

    def estimate_statistic(self, value: RealArray) -> RealArray:
        """
        Args:
            value: The value of the function output.
        """  # noqa: D205 D212
        if self.shape != value.shape:
            self.shape = value.shape
            self.reset(value.size)

        self._estimator.increment(value.ravel())
        return self._get_estimation().reshape(self.shape)

    def compute_jacobian(self, value: RealArray, jac_value: RealArray) -> RealArray:
        """
        Args:
            value: The value of the function output.
            jac_value: The value of the Jacobian.
        """  # noqa: D205 D212
        if self.jac_shape != jac_value.shape:
            self.jac_shape = jac_value.shape
            self.reset(value.size)

        self._jac_estimator.increment(jac_value.ravel())
        return self._get_estimation_jacobian().reshape(self.jac_shape)
