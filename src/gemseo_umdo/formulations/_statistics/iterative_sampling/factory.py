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

from gemseo.core.base_factory import BaseFactory

from gemseo_umdo.formulations._statistics.iterative_sampling.base_sampling_estimator import (  # noqa: E501
    BaseSamplingEstimator,
)


class SamplingEstimatorFactory(BaseFactory):
    """The factory of iterative sampling estimators."""

    _CLASS = BaseSamplingEstimator
    _PACKAGE_NAMES = ("gemseo_umdo.formulations._statistics.iterative_sampling",)
