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
"""Base statistic estimator for U-MDO formulations based on Taylor polynomials."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo_umdo.formulations.statistics.base_statistic_estimator import (
    BaseStatisticEstimator,
)

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace
    from numpy import ndarray
    from numpy.typing import NDArray


class TaylorPolynomialEstimator(BaseStatisticEstimator):
    """Base statistic estimator for a U-MDO formulation using Taylor polynomials."""

    _standard_deviations: NDArray[float]
    """The standard deviations associated with each component of the uncertain space."""

    def __init__(self, uncertain_space: ParameterSpace) -> None:
        """
        Args:
            uncertain_space: The uncertain variables
                with their probability distributions.
        """  # noqa: D205 D212 D415
        self._standard_deviations = uncertain_space.distribution.standard_deviation

    @abstractmethod
    def __call__(self, func: ndarray, jac: ndarray, hess: ndarray) -> ndarray:
        """
        Args:
            func: The output value at the mean value of the uncertain variables.
            jac: The Jacobian value at the mean value of the uncertain variables.
            hess: The Hessian value at the mean value of the uncertain variables.
        """  # noqa: D205 D212 D415 E112
