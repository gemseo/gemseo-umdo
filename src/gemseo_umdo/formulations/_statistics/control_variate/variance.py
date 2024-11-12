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

from gemseo_umdo.formulations._statistics.control_variate.base_control_variate_estimator import (  # noqa: E501
    BaseControlVariateEstimator,
)


class Variance(BaseControlVariateEstimator):
    """Estimator of the variance."""

    def estimate_statistic(  # noqa: D102
        self,
        samples: RealArray,
        u_samples: RealArray,
        mean: RealArray,
        jac: RealArray,
    ) -> RealArray:
        cv_samples = self._compute_control_variate_samples(u_samples, mean, jac)
        diff2 = (samples - samples.mean()) ** 2
        cv_diff2 = (cv_samples - cv_samples.mean()) ** 2
        cv_var = diagonal(multi_dot([jac, diag(self._u_standard_deviation**2), jac.T]))
        alpha = self._compute_opposite_scaled_covariance(diff2, cv_diff2)
        return (diff2.mean(0) + alpha * (cv_diff2.mean(0) - cv_var)).ravel()
