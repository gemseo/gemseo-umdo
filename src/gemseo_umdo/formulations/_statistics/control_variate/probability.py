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
"""Estimator of a probability for U-MDO formulations based on control variates."""

from __future__ import annotations

from operator import ge
from operator import le
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from gemseo_umdo.formulations._statistics.control_variate.base_control_variate_estimator import (  # noqa: E501
    BaseControlVariateEstimator,
)

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.typing import RealArray


class Probability(BaseControlVariateEstimator):
    """Estimator of a probability."""

    __compare: Callable[[Any, Any], Any]
    """The comparison operator."""

    __threshold: float
    """The threshold against which the probability is estimated."""

    def __init__(
        self,
        uncertain_space: ParameterSpace,
        threshold: float = 0.0,
        greater: bool = True,
    ) -> None:
        """
        Args:
            threshold: The threshold against which the probability is estimated.
            greater: Whether to compute the probability of exceeding the threshold.
        """  # noqa: D205 D212 D415
        super().__init__(uncertain_space)
        self.__threshold = threshold
        self.__compare = ge if greater else le

    def estimate_statistic(  # noqa: D102
        self,
        evaluations: RealArray,
        mean: RealArray,
        variance: RealArray,
        some_predictions: RealArray,
        many_predictions: RealArray,
    ) -> RealArray:
        evaluations = self.__compare(evaluations, self.__threshold)
        some_predictions = self.__compare(some_predictions, self.__threshold)
        many_predictions = self.__compare(many_predictions, self.__threshold)
        alpha = self._compute_opposite_scaled_covariance(evaluations, some_predictions)
        return evaluations.mean(0) + alpha * (
            some_predictions.mean(0) - many_predictions.mean(0)
        )
