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

    __n_samples: int
    """The sample size to approximate the statistic with the control variates."""

    def __init__(
        self,
        uncertain_space: ParameterSpace,
        threshold: float = 0.0,
        greater: bool = True,
        n_samples: int = 10000,
    ) -> None:
        """
        Args:
            threshold: The threshold against which the probability is estimated.
            greater: Whether to compute the probability of exceeding the threshold.
            n_samples: A high number of samples to approximate the statistic
                with the control variates.
        """  # noqa: D205 D212 D415
        super().__init__(uncertain_space)
        self.__threshold = threshold
        self.__compare = ge if greater else le
        self.__n_samples = n_samples

    def estimate_statistic(  # noqa: D102
        self,
        samples: RealArray,
        u_samples: RealArray,
        mean: RealArray,
        jac: RealArray,
    ) -> RealArray:
        cv_samples = self._compute_control_variate_samples(u_samples, mean, jac)
        ref_cv_samples = (
            mean + self._uncertain_space.compute_samples(self.__n_samples) @ jac.T
        )
        samples = self.__compare(samples, self.__threshold)
        cv_samples = self.__compare(cv_samples, self.__threshold)
        ref_cv_samples = self.__compare(ref_cv_samples, self.__threshold)
        alpha = self._compute_opposite_scaled_covariance(samples, cv_samples)
        return samples.mean(0) + alpha * (cv_samples.mean(0) - ref_cv_samples.mean(0))
