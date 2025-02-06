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
"""Estimators of a margin for a U-MDO formulation based on PCE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.pce.base_pce_estimator import BasePCEEstimator

if TYPE_CHECKING:
    from typing import ClassVar

    from gemseo.typing import RealArray


class Margin(BasePCEEstimator):
    """Estimator of a margin, i.e. mean + factor * deviation."""

    __factor: float
    """The factor related to the standard deviation."""

    ARG_NAMES: ClassVar[tuple[str]] = (
        BasePCEEstimator.MEAN_ARG_NAME,
        BasePCEEstimator.STD_ARG_NAME,
    )

    def __init__(self, factor: float = 2.0) -> None:
        """
        Args:
            factor: The factor related to the standard deviation.
        """  # noqa: D205 D212 D415
        self.__factor = factor

    def estimate_statistic(
        self, mean: RealArray, standard_deviation: RealArray
    ) -> RealArray:
        return mean + self.__factor * standard_deviation

    def compute_jacobian(
        self, mean_jac: RealArray, standard_deviation_jac: RealArray
    ) -> RealArray:
        return mean_jac + self.__factor * standard_deviation_jac
