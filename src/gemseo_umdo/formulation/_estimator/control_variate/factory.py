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
"""A factory of statistic estimators for U-MDO formulations based on control variate."""

from __future__ import annotations

from typing import ClassVar

from gemseo.core.base_factory import BaseFactory

from gemseo_umdo.formulation._estimator.control_variate.base import (
    BaseControlVariateEstimator,
)


class ControlVariateEstimatorFactory(BaseFactory[BaseControlVariateEstimator]):
    """The factory of statistic estimators based on control variates."""

    _class: ClassVar[type[BaseControlVariateEstimator]] = BaseControlVariateEstimator

    _package_names: ClassVar[tuple[str, ...]] = (
        "gemseo_umdo.formulation._estimator.control_variate",
    )
