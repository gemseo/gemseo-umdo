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
"""Estimator of the standard deviation for control variate-based U-MDO formulations."""

from numpy import ndarray

from gemseo_umdo.formulations.statistics.control_variate.variance import Variance


class StandardDeviation(Variance):
    """Estimator of the standard deviation."""

    def __call__(  # noqa: D102
        self, samples: ndarray, u_samples: ndarray, mean: ndarray, jac: ndarray
    ) -> ndarray:
        return super().__call__(samples, u_samples, mean, jac) ** 0.5
