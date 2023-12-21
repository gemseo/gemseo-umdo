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
"""Estimator of the expectation for U-MDO formulations based on Taylor polynomials."""

from __future__ import annotations

from numpy import array
from numpy import ndarray
from numpy.linalg import multi_dot

from gemseo_umdo.formulations.statistics.taylor_polynomial.taylor_polynomial_estimator import (  # noqa: E501
    TaylorPolynomialEstimator,
)


class Mean(TaylorPolynomialEstimator):
    """Estimator of the expectation."""

    def __call__(  # noqa: D102
        self, func: ndarray, jac: ndarray, hess: ndarray
    ) -> ndarray:
        if hess is None:
            return func

        std = self._standard_deviations
        return func + 0.5 * array([
            multi_dot([std, sub_hess, std]) for sub_hess in hess
        ])
