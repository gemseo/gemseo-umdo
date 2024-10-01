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
"""Estimator of a margin for U-MDO formulations based on control variates."""

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.typing import RealArray

from gemseo_umdo.formulations._statistics.control_variate.base_control_variate_estimator import (  # noqa: E501
    BaseControlVariateEstimator,
)
from gemseo_umdo.formulations._statistics.control_variate.mean import Mean
from gemseo_umdo.formulations._statistics.control_variate.standard_deviation import (
    StandardDeviation,
)


class Margin(BaseControlVariateEstimator):
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
        self,
        samples: RealArray,
        u_samples: RealArray,
        mean: RealArray,
        jac: RealArray,
    ) -> RealArray:
        m = self.__mean.estimate_statistic(samples, u_samples, mean, jac)
        s = self.__standard_deviation.estimate_statistic(samples, u_samples, mean, jac)
        return m + self.__factor * s
