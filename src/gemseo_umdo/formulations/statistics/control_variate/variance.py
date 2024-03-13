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

from gemseo.typing import RealArray
from numpy import diag
from numpy import diagonal
from numpy.linalg import multi_dot

from gemseo_umdo.formulations.statistics.control_variate.control_variate_estimator import (  # noqa: E501
    ControlVariateEstimator,
)


class Variance(ControlVariateEstimator):
    """Estimator of the variance."""

    def __call__(  # noqa: D102
        self, samples: RealArray, u_samples: RealArray, mean: RealArray, jac: RealArray
    ) -> RealArray:
        sample_lf, sample_hf = self._compute_lf_and_hf_samples(
            samples, u_samples, mean, jac
        )
        diff2_hf = (sample_hf - sample_hf.mean()) ** 2
        diff2_lf = (sample_lf - sample_lf.mean()) ** 2
        var_lf = diagonal(multi_dot([jac, diag(self._u_standard_deviation**2), jac.T]))
        alpha = self._compute_opposite_scaled_covariance(diff2_hf, diff2_lf)
        return (diff2_hf.mean(0) + alpha * (diff2_lf.mean(0) - var_lf)).ravel()
