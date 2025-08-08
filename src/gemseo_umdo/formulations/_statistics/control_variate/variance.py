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
"""Estimators of the variance for U-MDO formulation based on control variates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.control_variate.base_control_variate_estimator import (  # noqa: E501
    BaseControlVariateEstimator,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Variance(BaseControlVariateEstimator):
    """Estimator of the variance."""

    def estimate_statistic(  # noqa: D102
        self,
        evaluations: RealArray,
        mean: RealArray,
        variance: RealArray,
        some_predictions: RealArray,
        many_predictions: RealArray,
    ) -> RealArray:
        diff2 = (evaluations - evaluations.mean(0)) ** 2
        cv_diff2 = (some_predictions - some_predictions.mean(0)) ** 2
        alpha = self._compute_opposite_scaled_covariance(diff2, cv_diff2)
        return (diff2.mean(0) + alpha * (cv_diff2.mean(0) - variance)).ravel()
