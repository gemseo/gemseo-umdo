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
from numpy import nan_to_num
from numpy import newaxis

from gemseo_umdo.formulations._statistics.iterative_sampling.variance import Variance

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class StandardDeviation(Variance):
    """Iterative estimator of the standard deviation."""

    __var_jac: Variance
    """The iterative estimator of the Jacobian of the variance."""

    def __init__(self) -> None:
        super().__init__()
        self.__var_jac = Variance()

    def _get_estimation(self) -> RealArray:
        return array(self._estimator.getStandardDeviation())

    def compute_jacobian(self, value: RealArray, jac_value: RealArray) -> RealArray:
        self.jac_shape = jac_value.shape
        return nan_to_num(
            self.__var_jac.compute_jacobian(value, jac_value).reshape(self.jac_shape)
            / self._get_estimation()[:, newaxis]
            / 2
        ).ravel()

    def reset(self, size: int) -> None:
        super().reset(size)
        self.__var_jac.reset(size)
