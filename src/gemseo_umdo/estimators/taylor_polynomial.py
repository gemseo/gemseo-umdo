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
"""Estimators of statistic for U-MDO formulation based on Taylor polynomials."""
from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from gemseo.core.base_factory import BaseFactory
from numpy import array
from numpy import diag
from numpy import diagonal
from numpy.linalg import multi_dot

if TYPE_CHECKING:
    from gemseo_umdo.formulations.taylor_polynomial import TaylorPolynomial

from numpy import ndarray

from gemseo_umdo.estimators.estimator import BaseStatisticEstimator


class TaylorPolynomialEstimator(BaseStatisticEstimator):
    """Base statistic estimator for a U-MDO formulation using Taylor polynomials."""

    def __init__(self, formulation: TaylorPolynomial) -> None:  # noqa: D107
        super().__init__(formulation)


class TaylorPolynomialEstimatorFactory(BaseFactory):
    """The factory of :class:`.TaylorPolynomialEstimator`."""

    _CLASS = TaylorPolynomialEstimator
    _MODULE_NAMES = ()


class Mean(TaylorPolynomialEstimator):
    """Estimator of the expectation, a.k.a.

    mean.
    """

    def __call__(
        self, func: ndarray, jac: ndarray, hess: ndarray, **kwargs: Any
    ) -> float | ndarray:
        """
        Args:
            func: The output value at the mean value of the uncertain variables.
            jac: The Jacobian value at the mean value of the uncertain variables.
            hess: The Hessian value at the mean value of the uncertain variables.
        """  # noqa: D205 D212 D415
        if hess is None:
            return func

        std = self._formulation._uncertain_space.distribution.standard_deviation
        return func + 0.5 * array(
            [multi_dot([std, sub_hess, std]) for sub_hess in hess]
        )


class Variance(TaylorPolynomialEstimator):
    """Estimator of the variance."""

    def __call__(
        self, func: ndarray, jac: ndarray, hess: ndarray, **kwargs: Any
    ) -> float | ndarray:
        """
        Args:
            func: The output value at the mean value of the uncertain variables.
            jac: The Jacobian value at the mean value of the uncertain variables.
            hess: The Hessian value at the mean value of the uncertain variables.
        """  # noqa: D205 D212 D415
        std = self._formulation._uncertain_space.distribution.standard_deviation
        return diagonal(multi_dot([jac, diag(std**2), jac.T]))


class StandardDeviation(Variance):
    """Estimator of the standard deviation."""

    def __call__(
        self, func: ndarray, jac: ndarray, hess: ndarray, **kwargs: Any
    ) -> float | ndarray:
        """
        Args:
            func: The output value at the mean value of the uncertain variables.
            jac: The Jacobian value at the mean value of the uncertain variables.
            hess: The Hessian value at the mean value of the uncertain variables.
        """  # noqa: D205 D212 D415
        return super().__call__(func, jac, hess, **kwargs) ** 0.5


class Margin(TaylorPolynomialEstimator):
    """Estimator of a margin, i.e. mean + factor * deviation."""

    def __init__(self, formulation: TaylorPolynomial) -> None:  # noqa: D107
        super().__init__(formulation)
        self.__mean = Mean(formulation)
        self.__standard_deviation = StandardDeviation(formulation)

    def __call__(
        self,
        func: ndarray,
        jac: ndarray,
        hess: ndarray,
        factor: float = 2.0,
        **kwargs: Any,
    ) -> float | ndarray:
        """# noqa: D205 D212 D415
        Args:
            func: The output value at the mean value of the uncertain variables.
            jac: The Jacobian value at the mean value of the uncertain variables.
            hess: The Hessian value at the mean value of the uncertain variables.
            factor: The factor related to the standard deviation.
        """
        return self.__mean(
            func, jac, hess, **kwargs
        ) + factor * self.__standard_deviation(func, jac, hess, **kwargs)
