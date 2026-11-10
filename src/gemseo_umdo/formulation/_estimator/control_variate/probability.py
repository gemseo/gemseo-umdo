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

from gemseo_umdo.formulation._estimator.control_variate.base import (
    BaseControlVariateEstimator,
)
from gemseo_umdo.statistic.settings import Probability_Settings

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class Probability(BaseControlVariateEstimator[Probability_Settings]):
    """Estimator of a probability."""

    def estimate_statistic(  # noqa: D102
        self,
        evaluations: RealArray,
        mean: RealArray,
        variance: RealArray,
        some_predictions: RealArray,
        many_predictions: RealArray,
    ) -> RealArray:
        settings = self._settings
        compare = ge if settings.greater else le
        threshold = settings.threshold
        evaluations = compare(evaluations, threshold)
        some_predictions = compare(some_predictions, threshold)
        many_predictions = compare(many_predictions, threshold)
        alpha = self._compute_opposite_scaled_covariance(evaluations, some_predictions)
        return evaluations.mean(0) + alpha * (
            some_predictions.mean(0) - many_predictions.mean(0)
        )
