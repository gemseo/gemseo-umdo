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
"""Iterative estimator of a probability for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from openturns import GreaterOrEqual
from openturns import IterativeThresholdExceedance
from openturns import LessOrEqual

from gemseo_umdo.formulation._estimator.iterative_sampling.base import (
    BaseIterativeSamplingEstimator,
)
from gemseo_umdo.statistic.settings.probability import Probability_Settings

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class Probability(BaseIterativeSamplingEstimator[Probability_Settings]):
    """Iterative estimator of a probability."""

    can_compute_jacobian: ClassVar[bool] = False

    _estimator: IterativeThresholdExceedance | None

    def _get_estimation(self) -> RealArray:
        return array(
            self._estimator.getThresholdExceedance()
            / self._estimator.getIterationNumber()
        )

    def _create_estimator(self, size: int) -> None:
        settings = self._settings
        self._estimator = IterativeThresholdExceedance(
            size,
            GreaterOrEqual() if settings.greater else LessOrEqual(),
            settings.threshold,
        )
