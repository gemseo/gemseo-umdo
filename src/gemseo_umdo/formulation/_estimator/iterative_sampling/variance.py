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

from gemseo_umdo.formulation._estimator.iterative_sampling.base_central_moment import (  # noqa: E501
    BaseCentralMoment,
)
from gemseo_umdo.formulation._estimator.iterative_sampling.mean import Mean

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class Variance(BaseCentralMoment):
    """Iterative estimator of the variance."""

    _order: ClassVar[int] = 2

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

    @property
    def min_n_samples(self) -> int:  # noqa: D102
        return 2

    def _get_estimation(self) -> RealArray:
        return array(self._estimator.getVariance())

    def compute_jacobian(self, value: RealArray, jac_value: RealArray) -> RealArray:
        self.jac_shape = jac_value.shape
        # The means are updated first,
        # as their estimators do not exist before their first update,
        # e.g. when the Jacobian is estimated before the statistic.
        prod_mean = self.__prod_mean.estimate_statistic(value[:, newaxis] * jac_value)
        mean = self.__mean.estimate_statistic(value)
        mean_jac = self.__mean_jac.estimate_statistic(jac_value)
        n = self.__mean._estimator.getIterationNumber()
        alpha = n / (n - 1) if n > 1 else 1
        return 2 * (prod_mean - mean[:, newaxis] * mean_jac) * alpha

    def reset(self) -> None:  # noqa: D102
        super().reset()
        self.__mean_jac.reset()
        self.__mean.reset()
        self.__prod_mean.reset()
