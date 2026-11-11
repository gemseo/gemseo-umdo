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
"""The settings of the quantile."""

from __future__ import annotations

from decimal import Decimal
from typing import ClassVar
from typing import Self

from numpy import array
from numpy import quantile
from pydantic import Field
from pydantic import field_validator

from gemseo_umdo.statistic.settings.core.base import BaseStatisticSettings


class Quantile_Settings(BaseStatisticSettings):  # noqa: N801
    r"""The settings of the quantile."""

    _name_field_names: ClassVar[tuple[str, ...]] = ("level",)

    level: float = Field(
        0.95,
        gt=0,
        lt=1,
        description=r"""The level $\alpha$ of the quantile, in $]0,1[$.

For an objective or an inequality constraint,
it is replaced by $1-\alpha$ if necessary to make the quantile conservative.""",
    )

    method: str = Field(
        "linear",
        description="""The name of the method of
[numpy.quantile][numpy.quantile] to estimate the quantile.""",
    )

    @field_validator("method")
    @classmethod
    def _check_method(cls, method: str) -> str:
        """Check that the method is a method of `numpy.quantile`.

        Args:
            method: The name of the method.

        Returns:
            The name of the method.

        Raises:
            ValueError: When the method is not a method of `numpy.quantile`.
        """
        # The method is checked with a cheap call to numpy.quantile
        # so that a wrong method fails before any sampling,
        # as NumPy does not expose its list of methods.
        try:
            quantile(array([0.0, 1.0]), 0.5, method=method)
        except ValueError as error:
            msg = (
                "The method of the quantile must be a method of numpy.quantile; "
                f"got '{method}'."
            )
            raise ValueError(msg) from error

        return method

    def make_conservative(self, from_below: bool) -> Self:  # noqa: D102
        # The conversion to float handles NumPy scalars,
        # whose repr is not a valid decimal string, e.g. np.float64(0.9).
        level = float(self.level)
        # The decimal arithmetic avoids names like Q[f; 0.09999999999999998]
        # for the complementary level of 0.9.
        complementary_level = float(1 - Decimal(repr(level)))
        level = (
            min(level, complementary_level)
            if from_below
            else max(level, complementary_level)
        )
        return self.model_copy(update={"level": level})
