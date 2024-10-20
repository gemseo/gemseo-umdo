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
"""Base statistic estimator for sampling-based U-MDO formulations."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo_umdo.formulations._statistics.base_statistic_estimator import (
    BaseStatisticEstimator,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class BaseSamplingEstimator(BaseStatisticEstimator):
    """Base statistic estimator for a U-MDO formulation using sampling."""

    @abstractmethod
    def estimate_statistic(self, samples: RealArray) -> RealArray:
        """
        Args:
            samples: The samples to estimate the statistic.
        """  # noqa: D205 D212

    def compute_jacobian(self, samples: RealArray, jac_samples: RealArray) -> RealArray:
        """
        Args:
            samples: The samples to estimate the statistic.
            jac_samples: The samples to estimate the Jacobian of the statistic.
        """  # noqa: D205 D212
        raise NotImplementedError
