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
"""The settings of the margin."""

from __future__ import annotations

from typing import ClassVar
from typing import Self

from pydantic import Field
from pydantic import field_validator

from gemseo_umdo.statistic.settings.core.base import BaseStatisticSettings


class Margin_Settings(BaseStatisticSettings):  # noqa: N801
    r"""The settings of the margin $\mathbb{E}[\varphi]+\kappa\mathbb{S}[\varphi]$."""

    _name_field_names: ClassVar[tuple[str, ...]] = ("factor",)

    factor: float = Field(
        2.0,
        description=r"""The factor $\kappa$ related to the standard deviation.

For an objective or an inequality constraint,
its sign is changed if necessary to make the margin conservative.""",
    )

    @field_validator("factor")
    @classmethod
    def _check_factor(cls, factor: float) -> float:
        """Check that the factor is non-zero.

        Args:
            factor: The factor.

        Returns:
            The factor.

        Raises:
            ValueError: When the factor is zero.
        """
        if factor == 0:
            msg = (
                "The factor of a margin must be non-zero; "
                "a margin with a zero factor is the mean. "
                "Use the statistic 'Mean' instead."
            )
            raise ValueError(msg)

        return factor

    def make_conservative(self, from_below: bool) -> Self:  # noqa: D102
        factor = abs(self.factor)
        if from_below:
            factor *= -1

        return self.model_copy(update={"factor": factor})
