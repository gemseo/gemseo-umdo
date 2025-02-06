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
"""Estimator of the variance for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import atleast_1d
from numpy import newaxis

from gemseo_umdo.formulations._statistics.sampling.base_sampling_estimator import (
    BaseSamplingEstimator,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Variance(BaseSamplingEstimator):
    """Estimator of the variance."""

    def estimate_statistic(self, samples: RealArray) -> RealArray:
        return atleast_1d(samples.var(0, ddof=1))

    def compute_jacobian(self, samples: RealArray, jac_samples: RealArray) -> RealArray:
        n = len(samples)
        return (
            2
            * n
            / (n - 1)
            * (
                (samples[..., newaxis] * jac_samples).mean(0)
                - jac_samples.mean(0) * samples.mean(0)[..., newaxis]
            )
        )
