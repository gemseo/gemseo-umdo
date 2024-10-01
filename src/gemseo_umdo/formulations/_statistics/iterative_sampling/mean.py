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

from numpy import array

from gemseo_umdo.formulations._statistics.iterative_sampling.base_central_moment import (  # noqa: E501
    BaseCentralMoment,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Mean(BaseCentralMoment):
    """Iterative estimator of the expectation."""

    _ORDER: ClassVar[int] = 1

    def _get_estimation(self) -> RealArray:
        return array(self._estimator.getMean())

    def _get_estimation_jacobian(self) -> RealArray:
        return array(self._jac_estimator.getMean())
