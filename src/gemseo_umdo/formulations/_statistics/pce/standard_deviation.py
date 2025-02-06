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
"""Estimator of the standard deviation for a U-MDO formulations based on PCE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.pce.base_pce_estimator import BasePCEEstimator

if TYPE_CHECKING:
    from typing import ClassVar

    from gemseo.typing import RealArray


class StandardDeviation(BasePCEEstimator):
    """Estimator of the standard deviation."""

    ARG_NAMES: ClassVar[tuple[str]] = (BasePCEEstimator.STD_ARG_NAME,)

    def estimate_statistic(self, standard_deviation: RealArray) -> RealArray:
        return standard_deviation

    def compute_jacobian(self, standard_deviation_jac: RealArray) -> RealArray:
        return standard_deviation_jac
