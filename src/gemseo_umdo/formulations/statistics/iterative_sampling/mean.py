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
"""Iterative estimator of the expectation for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo_umdo.formulations.statistics.iterative_sampling.central_moment import (
    CentralMoment,
)

if TYPE_CHECKING:
    from openturns import Point


class Mean(CentralMoment):
    """Iterative estimator of the expectation.

    This class iteratively computes the mean of an increasing dataset without storing
    any data in memory.
    """

    _ORDER: ClassVar[int] = 1

    def _get_central_moment(self) -> Point:
        return self._estimator.getMean()
