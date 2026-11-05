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
"""Iterative estimator of a moment for sampling-based U-MDO formulations."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar

from openturns import IterativeMoments

from gemseo_umdo.formulation._estimator.iterative_sampling.base import (  # noqa: E501
    BaseSamplingEstimator,
)

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class BaseCentralMoment(BaseSamplingEstimator):
    """Base iterative estimator of a central moment, e.g. expectation or variance."""

    _estimator: IterativeMoments | None
    _jac_estimator: IterativeMoments | None

    _order: ClassVar[int]
    """The order of the central moment."""

    def _create_estimator(self, size: int) -> None:
        self._estimator = IterativeMoments(self._order, size)

    def _create_jac_estimator(self, size: int) -> None:
        self._jac_estimator = IterativeMoments(self._order, size)

    @abstractmethod
    def _get_estimation(self) -> RealArray:
        """Return the current estimation of the central moment.

        Returns:
            The current estimation of the central moment.
        """
