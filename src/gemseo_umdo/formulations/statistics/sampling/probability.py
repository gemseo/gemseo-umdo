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
# Copyright 2021 IRT Saint Exupéry,7 https://www.irt-saintexupery.com
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
"""Estimator of a probability for sampling-based U-MDO formulations."""

from __future__ import annotations

from operator import ge
from operator import le
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from gemseo_umdo.formulations.statistics.sampling.sampling_estimator import (
    SamplingEstimator,
)

if TYPE_CHECKING:
    from numpy import ndarray


class Probability(SamplingEstimator):
    """Estimator of a probability."""

    __compare: Callable[[Any, Any], Any]
    """The comparison operator."""

    __threshold: float
    """The threshold against which the probability is estimated."""

    def __init__(self, threshold: float = 0.0, greater: bool = True) -> None:
        """
        Args:
            threshold: The threshold against which the probability is estimated.
            greater: Whether to compute the probability of exceeding the threshold.
        """  # noqa: D205 D212 D415
        super().__init__()
        self.__threshold = threshold
        self.__compare = ge if greater else le

    def __call__(self, samples: ndarray) -> ndarray:  # noqa: D102
        return self.__compare(samples, self.__threshold).mean(0)
