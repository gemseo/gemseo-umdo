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
"""Iterative estimator of a standard deviation for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import newaxis
from numpy import where

from gemseo_umdo.formulations._statistics.iterative_sampling.variance import Variance

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class StandardDeviation(Variance):
    """Iterative estimator of the standard deviation."""

    __estimated_since_jacobian: bool
    """Whether the statistic has been estimated since the last Jacobian estimation.

    It is independent of the samples whose Jacobian is not estimated,
    e.g. when the Jacobian of the output is missing for a sample.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__estimated_since_jacobian = False

    def _get_estimation(self) -> RealArray:
        return array(self._estimator.getStandardDeviation())

    def estimate_statistic(self, value: RealArray) -> RealArray:
        estimation = super().estimate_statistic(value)
        self.__estimated_since_jacobian = True
        return estimation

    def compute_jacobian(self, value: RealArray, jac_value: RealArray) -> RealArray:
        """
        Args:
            value: The value of the function output.
                `estimate_statistic(value)` must be called just before,
                as the Jacobian uses the standard deviation
                estimated from the samples up to `value` included.
            jac_value: The value of the Jacobian.

        Raises:
            ValueError: When `estimate_statistic(value)`
                has not been called just before.
        """  # noqa: D205 D212
        if not self.__estimated_since_jacobian:
            msg = (
                "The standard deviation must be estimated from the output value "
                "before its Jacobian."
            )
            raise ValueError(msg)

        self.__estimated_since_jacobian = False
        std = self._get_estimation()
        std_zeros = where(std == 0)[0]
        std[std == 0] = 1
        jac = super().compute_jacobian(value, jac_value) / std[:, newaxis] / 2
        jac[std_zeros] = 0
        return jac

    def reset(self) -> None:  # noqa: D102
        super().reset()
        self.__estimated_since_jacobian = False
