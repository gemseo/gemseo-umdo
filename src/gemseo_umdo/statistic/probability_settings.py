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
"""The settings of the probability."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from gemseo_umdo.statistic.core.base_settings import BaseStatisticSettings


class Probability_Settings(BaseStatisticSettings):  # noqa: N801
    """The settings of the probability that the output exceeds a threshold."""

    _name_field_names: ClassVar[tuple[str, ...]] = ("threshold",)

    threshold: float = Field(
        0.0, description="The threshold against which the probability is estimated."
    )

    greater: bool = Field(
        True,
        description="""Whether to compute the probability of exceeding the threshold.

Otherwise, compute the probability of being below the threshold.""",
    )
