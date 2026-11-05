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
"""Mixin for the estimators of a margin."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

if TYPE_CHECKING:
    from collections.abc import MutableMapping

_factor_name: Final[str] = "factor"
"""The name of the parameter for the factor related to the standard deviation."""


class BaseMargin:
    """Mixin for the estimators of a margin, i.e. mean + factor * deviation.

    It makes the factor conservative according to the use of the margin.
    """

    default_factor: ClassVar[float] = 2.0
    """The default factor related to the standard deviation."""

    @classmethod
    def update_parameters(
        cls, parameters: MutableMapping[str, Any], conservative_from_below: bool | None
    ) -> MutableMapping[str, Any]:
        """Update the parameters of the statistic according to its use.

        Args:
            parameters: The parameters of the statistic to be updated.
            conservative_from_below: Whether the statistic must be conservative
                from below, e.g. with a negative factor for a margin.
                If `None`,
                e.g. for an observable,
                the statistic does not have to be conservative.

        Returns:
            The updated parameters of the statistic.

        Raises:
            ValueError: When the factor is zero.
        """
        # The default factor is filled in so that the name of the statistic includes it.
        factor = parameters.get(_factor_name, cls.default_factor)
        if factor == 0:
            msg = (
                "The factor of a margin must be non-zero; "
                "a margin with a zero factor is the mean. "
                "Use the statistic 'Mean' instead."
            )
            raise ValueError(msg)

        if conservative_from_below is not None:
            factor = abs(factor)
            if conservative_from_below:
                factor *= -1

        parameters[_factor_name] = factor

        return parameters
