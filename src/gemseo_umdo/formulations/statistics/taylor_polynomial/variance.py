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
"""Estimator of the variance for U-MDO formulations based on Taylor polynomials."""

from __future__ import annotations

from numpy import diag
from numpy import diagonal
from numpy import ndarray
from numpy.linalg import multi_dot

from gemseo_umdo.formulations.statistics.taylor_polynomial.taylor_polynomial_estimator import (  # noqa: E501
    TaylorPolynomialEstimator,
)


class Variance(TaylorPolynomialEstimator):
    """Estimator of the variance."""

    def __call__(self, func: ndarray, jac: ndarray, hess: ndarray) -> ndarray:
        """
        Args:
            func: The output value at the mean value of the uncertain variables.
            jac: The Jacobian value at the mean value of the uncertain variables.
            hess: The Hessian value at the mean value of the uncertain variables.
        """  # noqa: D205 D212 D415
        return diagonal(multi_dot([jac, diag(self._standard_deviations**2), jac.T]))
