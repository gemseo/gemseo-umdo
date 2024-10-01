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
"""Base statistic estimator for a U-MDO formulations based on PCE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
    BaseStatisticEstimator,
)

if TYPE_CHECKING:
    from typing import ClassVar
    from typing import Final


class BasePCEEstimator(BaseStatisticEstimator):
    """Base statistic estimator for a U-MDO formulation using PCE."""

    MEAN_ARG_NAME: Final[str] = "mean"
    """The name of the mean argument."""

    STD_ARG_NAME: Final[str] = "std"
    """The name of the standard deviation argument."""

    VAR_ARG_NAME: Final[str] = "var"
    """The name of the variance argument."""

    ARG_NAMES: ClassVar[tuple[str]] = ()
    """The names of the arguments to be used for the estimator."""
