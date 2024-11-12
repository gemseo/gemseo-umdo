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
"""Estimators of a margin for U-MDO formulation based on Taylor polynomials."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.taylor_polynomial.base_taylor_polynomial_estimator import (  # noqa: E501
    BaseTaylorPolynomialEstimator,
)
from gemseo_umdo.formulations._statistics.taylor_polynomial.mean import Mean
from gemseo_umdo.formulations._statistics.taylor_polynomial.standard_deviation import (
    StandardDeviation,
)

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.typing import RealArray


class Margin(BaseTaylorPolynomialEstimator):
    """Estimator of a margin, i.e. mean + factor * deviation."""

    __factor: float
    """The factor related to the standard deviation."""

    __mean: Mean
    """The iterative estimator of the mean."""

    __standard_deviation: StandardDeviation
    """The iterative estimator of the standard deviation."""

    def __init__(self, uncertain_space: ParameterSpace, factor: float = 2.0) -> None:
        """
        Args:
            factor: The factor related to the standard deviation.
        """  # noqa: D205 D212 D415
        super().__init__(uncertain_space)
        self.__mean = Mean(uncertain_space)
        self.__standard_deviation = StandardDeviation(uncertain_space)
        self.__factor = factor

    def estimate_statistic(  # noqa: D102
        self, func: RealArray, jac: RealArray, hess: RealArray
    ) -> RealArray:
        mean = self.__mean.estimate_statistic(func, jac, hess)
        std = self.__standard_deviation.estimate_statistic(func, jac, hess)
        return mean + self.__factor * std
