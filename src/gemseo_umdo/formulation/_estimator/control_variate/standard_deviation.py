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

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulation._estimator.control_variate.variance import Variance
from gemseo_umdo.statistic.standard_deviation_settings import StandardDeviation_Settings

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class StandardDeviation(Variance[StandardDeviation_Settings]):
    """Estimator of the standard deviation."""

    def estimate_statistic(  # noqa: D102
        self,
        evaluations: RealArray,
        mean: RealArray,
        variance: RealArray,
        some_predictions: RealArray,
        many_predictions: RealArray,
    ) -> RealArray:
        return (
            super().estimate_statistic(
                evaluations, mean, variance, some_predictions, many_predictions
            )
            ** 0.5
        )
