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
"""Base estimator of statistic associated with a U-MDO formulation."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar

from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

from gemseo_umdo.formulations.statistic_settings import BaseStatisticSettings

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


T = TypeVar("T", bound=BaseStatisticSettings)


class BaseStatisticEstimator(Generic[T], metaclass=ABCGoogleDocstringInheritanceMeta):
    """The base statistic estimator for U-MDO formulations."""

    _settings: T
    """The settings of the statistic."""

    def __init__(self, settings: T) -> None:
        """
        Args:
            settings: The settings of the statistic.
        """  # noqa: D205 D212 D415
        self._settings = settings

    @property
    def min_n_samples(self) -> int:
        """The minimum number of samples to estimate the statistic and its Jacobian.

        This number is used by the sampling-based U-MDO formulations.
        """
        return 1

    @abstractmethod
    def estimate_statistic(self, *args: Any, **kwargs: Any) -> RealArray:  # noqa: D102
        """Estimate the statistic.

        Args:
            *args: The mandatory arguments.
            **kwargs: The optional arguments.

        Returns:
            The estimation of the statistic.
        """

    def compute_jacobian(self, *args: Any, **kwargs: Any) -> RealArray:  # noqa: D102
        """Compute the Jacobian of the statistic estimation.

        Args:
            *args: The mandatory arguments.
            **kwargs: The optional arguments.

        Returns:
            The Jacobian of the statistic estimation.
        """
        raise NotImplementedError
