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

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import TYPE_CHECKING

from gemseo.core.base_factory import BaseFactory
from numpy import ndarray

if TYPE_CHECKING:
    from gemseo_umdo.formulations.formulation import UMDOFormulation


class BaseStatisticEstimator(ABC):
    """The base estimator of statistics associated with a U-MDO formulation."""

    _formulation: UMDOFormulation
    """The U-MDO formulation."""

    def __init__(self, formulation: UMDOFormulation) -> None:
        """
        Args:
            formulation: The U-MDO formulation.
        """  # noqa: D205 D212 D415
        self._formulation = formulation

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> float | ndarray:  # noqa: D102
        ...


class BaseStatisticEstimatorFactory(BaseFactory):
    """The factory of :class:`.BaseStatisticEstimator`."""

    _CLASS = BaseStatisticEstimator
    _MODULE_NAMES = ("gemseo_umdo.estimators",)

    def create(
        self,
        name: str,
        formulation: UMDOFormulation,
        **options: Any,
    ) -> BaseStatisticEstimator:
        """Create a statistic estimator.

        Args:
            name: The class name of the statistic estimator.
            formulation: The U-MDO formulation.
            **options: The options of the statistic estimator.
        """
        return self.create(name, formulation=formulation, **options)
