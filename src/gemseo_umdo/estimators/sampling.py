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
"""Estimators of statistic for sampling-based U-MDO formulation."""
from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from gemseo.core.base_factory import BaseFactory

if TYPE_CHECKING:
    from gemseo_umdo.formulations.sampling import Sampling

from numpy import ndarray

from gemseo_umdo.estimators.estimator import BaseStatisticEstimator


class SamplingEstimator(BaseStatisticEstimator):
    """Base statistic estimator for a U-MDO formulation using sampling."""

    def __init__(self, formulation: Sampling) -> None:  # noqa: D107
        super().__init__(formulation)


class SamplingEstimatorFactory(BaseFactory):
    """The factory of sampling-based statistic estimators."""

    _CLASS = SamplingEstimator
    _MODULE_NAMES = ()


class Mean(SamplingEstimator):
    """Estimator of the expectation, a.k.a.

    mean.
    """

    def __call__(self, samples: ndarray, **kwargs: Any) -> float | ndarray:
        """
        Args:
            samples: The output evaluations arranged in rows.
        """  # noqa: D205 D212 D415
        return samples.mean(0)


class Variance(SamplingEstimator):
    """Estimator of the variance."""

    def __call__(self, samples: ndarray, **kwargs: Any) -> float | ndarray:
        """
        Args:
            samples: The output evaluations arranged in rows.
        """  # noqa: D205 D212 D415
        return samples.var(0)


class Probability(SamplingEstimator):
    """Estimator of a probability."""

    def __call__(
        self,
        samples: ndarray,
        threshold: float,
        greater: bool = True,
        **kwargs: Any,
    ) -> float | ndarray:
        """
        Args:
            samples: The output evaluations arranged in rows.
            threshold: The threshold against which the probability is estimated.
            greater: Whether to compute the probability of exceeding the threshold.
        """  # noqa: D205 D212 D415
        if greater:
            return (samples >= threshold).mean(0)
        else:
            return (samples <= threshold).mean(0)


class StandardDeviation(Variance):
    """Estimator of the standard deviation."""

    def __call__(self, samples: ndarray, **kwargs: Any) -> float | ndarray:
        """
        Args:
            samples: The output evaluations arranged in rows.
        """  # noqa: D205 D212 D415
        return super().__call__(samples, **kwargs) ** 0.5


class Margin(SamplingEstimator):
    """Estimator of a margin, i.e. mean + factor * deviation."""

    def __init__(self, formulation: Sampling) -> None:  # noqa: D107
        super().__init__(formulation)
        self.__mean = Mean(formulation)
        self.__standard_deviation = StandardDeviation(formulation)

    def __call__(
        self,
        samples: ndarray,
        factor: float = 2.0,
        **kwargs: Any,
    ) -> float | ndarray:
        """
        Args:
            samples: The output evaluations arranged in rows.
            factor: The factor related to the standard deviation.
        """  # noqa: D205 D212 D415
        return self.__mean(samples, **kwargs) + factor * self.__standard_deviation(
            samples, **kwargs
        )
