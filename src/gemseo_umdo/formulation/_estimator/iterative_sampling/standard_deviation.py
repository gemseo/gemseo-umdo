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

from gemseo_umdo.formulation._estimator.iterative_sampling.variance import Variance
from gemseo_umdo.statistic.settings import StandardDeviation_Settings

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class StandardDeviation(Variance[StandardDeviation_Settings]):
    """Iterative estimator of the standard deviation."""

    def _get_estimation(self) -> RealArray:
        return array(self._estimator.getStandardDeviation())

    def compute_jacobian(self, value: RealArray, jac_value: RealArray) -> RealArray:
        self.estimate_statistic(value)
        std = self._get_estimation()
        std_zeros = where(std == 0)[0]
        std[std == 0] = 1
        jac = super().compute_jacobian(value, jac_value) / std[:, newaxis] / 2
        jac[std_zeros] = 0
        return jac
