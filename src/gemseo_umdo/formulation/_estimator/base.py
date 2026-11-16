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
from typing import ClassVar

from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from gemseo.util.typing import RealArray


class BaseStatisticEstimator(metaclass=ABCGoogleDocstringInheritanceMeta):
    """The base statistic estimator for U-MDO formulations."""

    can_compute_jacobian: ClassVar[bool] = False
    """Whether the estimator can estimate the Jacobian of the statistic."""

    @property
    def min_n_samples(self) -> int:
        """The minimum number of samples to estimate the statistic and its Jacobian.

        This number is used by the sampling-based U-MDO formulations.
        """
        return 1

    @classmethod
    def update_parameters(
        cls, parameters: MutableMapping[str, Any], conservative_from_below: bool | None
    ) -> MutableMapping[str, Any]:
        """Update the parameters of the statistic according to its use.

        By default, the parameters are left unchanged.

        Args:
            parameters: The parameters of the statistic to be updated.
            conservative_from_below: Whether the statistic must be conservative
                from below,
                e.g. with a negative factor for a margin
                and a level lower than 0.5 for a quantile.
                If `None`,
                e.g. for an observable,
                the statistic does not have to be conservative.

        Returns:
            The updated parameters of the statistic.
        """
        return parameters

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
