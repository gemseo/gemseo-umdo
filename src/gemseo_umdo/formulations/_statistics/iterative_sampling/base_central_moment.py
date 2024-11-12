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

from typing import ClassVar

from openturns import IterativeMoments

from gemseo_umdo.formulations._statistics.iterative_sampling.base_sampling_estimator import (  # noqa: E501
    BaseSamplingEstimator,
)


class BaseCentralMoment(BaseSamplingEstimator):
    """Base iterative estimator of a central moment, e.g. expectation or variance."""

    _estimator: IterativeMoments | None
    _jac_estimator: IterativeMoments | None

    _ORDER: ClassVar[int]
    """The order of the central moment."""

    def reset(self, size: int) -> None:  # noqa: D102
        super().reset(size)
        self._estimator = IterativeMoments(self._ORDER, size)
        if self.jac_shape is not None:
            self._jac_estimator = IterativeMoments(
                self._ORDER, self.jac_shape[0] * self.jac_shape[1]
            )
