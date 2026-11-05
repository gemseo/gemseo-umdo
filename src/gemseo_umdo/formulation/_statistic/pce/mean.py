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
"""Estimator of the expectation for a U-MDO formulations based on PCE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulation._statistic.pce.base_pce_estimator import BasePCEEstimator

if TYPE_CHECKING:
    from typing import ClassVar

    from gemseo.util.typing import RealArray


class Mean(BasePCEEstimator):
    """Estimator of the expectation."""

    arg_names: ClassVar[tuple[str]] = (BasePCEEstimator.mean_arg_name,)

    def estimate_statistic(self, mean: RealArray) -> RealArray:
        return mean

    def compute_jacobian(self, mean_jac: RealArray) -> RealArray:
        return mean_jac
