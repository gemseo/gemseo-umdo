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
"""A function to compute a statistic from `Sampling`.

See also [Sampling][gemseo_umdo.formulations.sampling.Sampling].
"""

from __future__ import annotations

from gemseo_umdo.formulations._functions.base_statistic_function import (
    BaseStatisticFunction,
)
from gemseo_umdo.formulations._functions.base_statistic_function import UMDOFormulationT


class BaseStatisticFunctionForSampling(BaseStatisticFunction[UMDOFormulationT]):
    """A function to compute a statistic from `Sampling`."""

    @property
    def _observable_name(self) -> str:
        """The name of the observable."""
        return f"{self._estimate_statistic.__class__.__name__}[{self._function_name}]"
