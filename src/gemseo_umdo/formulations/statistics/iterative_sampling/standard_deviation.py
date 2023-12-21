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
"""Iterative estimator of a standard deviation for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_umdo.formulations.statistics.iterative_sampling.variance import Variance

if TYPE_CHECKING:
    from openturns import Point


class StandardDeviation(Variance):
    """Iterative estimator of the standard deviation.

    This class iteratively computes the standard deviation of an increasing dataset
    without storing any data in memory.
    """

    def _get_statistic(self) -> Point:
        return self._estimator.getStandardDeviation()
