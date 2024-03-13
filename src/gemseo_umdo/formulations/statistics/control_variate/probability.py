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

from gemseo_umdo.formulations.statistics.control_variate.control_variate_estimator import (  # noqa: E501
    ControlVariateEstimator,
)

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.typing import RealArray


class Probability(ControlVariateEstimator):
    """Estimator of a probability."""

    __compare: Callable[[Any, Any], Any]
    """The comparison operator."""

    __threshold: float
    """The threshold against which the probability is estimated."""

    __n_samples: int
    """The sample size to approximate the statistic with the low-fidelity model."""

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
                with the low-fidelity model.
        """  # noqa: D205 D212 D415
        super().__init__(uncertain_space)
        self.__threshold = threshold
        self.__compare = ge if greater else le
        self.__n_samples = n_samples

    def __call__(  # noqa: D102
        self, samples: RealArray, u_samples: RealArray, mean: RealArray, jac: RealArray
    ) -> RealArray:
        sample_lf, sample_hf = self._compute_lf_and_hf_samples(
            samples, u_samples, mean, jac
        )
        ref_sample_lf = (
            mean + self._uncertain_space.compute_samples(self.__n_samples) @ jac.T
        )
        sample_lf = self.__compare(sample_lf, self.__threshold)
        sample_hf = self.__compare(sample_hf, self.__threshold)
        ref_sample_lf = self.__compare(ref_sample_lf, self.__threshold)
        alpha = self._compute_opposite_scaled_covariance(sample_hf, sample_lf)
        return sample_hf.mean(0) + alpha * (sample_lf.mean(0) - ref_sample_lf.mean(0))
