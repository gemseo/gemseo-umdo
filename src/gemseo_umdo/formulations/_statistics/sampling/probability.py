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
"""Estimator of a probability for sampling-based U-MDO formulations."""

from __future__ import annotations

from operator import ge
from operator import le
from typing import TYPE_CHECKING
from typing import Any

from gemseo_umdo.formulations._statistics.sampling.base_sampling_estimator import (
    BaseSamplingEstimator,
)
from gemseo_umdo.formulations.statistic_settings import Probability_Settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.util.typing import RealArray


class Probability(BaseSamplingEstimator[Probability_Settings]):
    """Estimator of a probability."""

    __compare: Callable[[Any, Any], Any]
    """The comparison operator."""

    def __init__(self, settings: Probability_Settings) -> None:
        """
        Args:
            settings: The settings of the probability.
        """  # noqa: D205 D212 D415
        super().__init__(settings)
        self.__compare = ge if settings.greater else le

    def estimate_statistic(self, samples: RealArray) -> RealArray:
        return self.__compare(samples, self._settings.threshold).mean(0)
