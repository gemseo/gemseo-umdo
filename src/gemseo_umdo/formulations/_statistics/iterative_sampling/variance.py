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
"""Iterative estimator of the variance for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from numpy import newaxis

from gemseo_umdo.formulations._statistics.iterative_sampling.base_central_moment import (  # noqa: E501
    BaseCentralMoment,
)
from gemseo_umdo.formulations._statistics.iterative_sampling.mean import Mean

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Variance(BaseCentralMoment):
    """Iterative estimator of the variance."""

    _ORDER: ClassVar[int] = 2

    __mean_jac: Mean
    """The iterative estimator of the Jacobian of the mean."""

    __mean: Mean
    """The iterative estimator of the mean."""

    __prod_mean: Mean
    """The iterative estimator of the product between the mean and its Jacobian."""

    def __init__(self) -> None:
        super().__init__()
        self.__mean_jac = Mean()
        self.__mean = Mean()
        self.__prod_mean = Mean()

    def _get_estimation(self) -> RealArray:
        return array(self._estimator.getVariance())

    def compute_jacobian(self, value: RealArray, jac_value: RealArray) -> RealArray:
        self.jac_shape = jac_value.shape
        n = self.__mean._estimator.getIterationNumber() + 1
        alpha = n / (n - 1) if n > 1 else 1
        return (
            2
            * (
                self.__prod_mean.estimate_statistic(value[:, newaxis] * jac_value)
                - (
                    self.__mean.estimate_statistic(value)[:, newaxis]
                    * self.__mean_jac.estimate_statistic(jac_value)
                )
            )
            * alpha
        )

    def reset(self, size: int) -> None:  # noqa: D102
        super().reset(size)
        self.__mean_jac.reset(size)
        self.__mean.reset(size)
        self.__prod_mean.reset(size)
