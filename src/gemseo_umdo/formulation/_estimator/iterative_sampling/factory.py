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
"""A factory of iterative statistic estimators for sampling-based U-MDO formulations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.base_factory import BaseFactory

from gemseo_umdo.formulation._estimator.iterative_sampling.base import (
    BaseSamplingEstimator,
)

if TYPE_CHECKING:
    from gemseo.core.base_factory import _ClassInfo


class SamplingEstimatorFactory(BaseFactory[BaseSamplingEstimator]):
    """The factory of iterative sampling estimators."""

    _class: ClassVar[type[BaseSamplingEstimator]] = BaseSamplingEstimator

    _package_names: ClassVar[tuple[str, ...]] = (
        "gemseo_umdo.formulation._estimator.iterative_sampling",
    )

    @property
    def _name_to_class_info(self) -> dict[str, _ClassInfo[BaseSamplingEstimator]]:
        # The base class is not abstract but does not estimate any statistic.
        # The dictionary cached by the base factory is filtered into a new one,
        # so that reading this property does not alter this cache.
        base_class_name = self._class.__name__
        return {
            name: class_info
            for name, class_info in super()._name_to_class_info.items()
            if name != base_class_name
        }
