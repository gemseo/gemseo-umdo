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

from gemseo_umdo.formulations.statistics.control_variate.control_variate_estimator import (  # noqa: E501
    ControlVariateEstimator,
)

if TYPE_CHECKING:
    from numpy import ndarray


class Mean(ControlVariateEstimator):
    """Estimator of the expectation."""

    def __call__(  # noqa: D102
        self, samples: ndarray, u_samples: ndarray, mean: ndarray, jac: ndarray
    ) -> ndarray:
        sample_lf, sample_hf = self._compute_lf_and_hf_samples(
            samples, u_samples, mean, jac
        )
        alpha = self._compute_opposite_scaled_covariance(sample_hf, sample_lf)
        return samples.mean(0) + alpha * (sample_lf.mean(0) - mean)
