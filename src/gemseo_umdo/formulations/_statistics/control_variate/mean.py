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
"""Estimator of the expectation for U-MDO formulations based on control variates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.control_variate.base_control_variate_estimator import (  # noqa: E501
    BaseControlVariateEstimator,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Mean(BaseControlVariateEstimator):
    """Estimator of the expectation."""

    def estimate_statistic(  # noqa: D102
        self,
        samples: RealArray,
        u_samples: RealArray,
        mean: RealArray,
        jac: RealArray,
    ) -> RealArray:
        cv_samples = self._compute_control_variate_samples(u_samples, mean, jac)
        alpha = self._compute_opposite_scaled_covariance(samples, cv_samples)
        return samples.mean(0) + alpha * (cv_samples.mean(0) - mean)
