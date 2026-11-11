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

from typing import TYPE_CHECKING

from gemseo_umdo.formulation._estimator.sampling.base import BaseSamplingEstimator
from gemseo_umdo.statistic.settings.probability import Probability_Settings

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class Probability(BaseSamplingEstimator[Probability_Settings]):
    """Estimator of a probability."""

    def estimate_statistic(self, samples: RealArray) -> RealArray:
        settings = self._settings
        return settings.comparison_operator(samples, settings.threshold).mean(0)
