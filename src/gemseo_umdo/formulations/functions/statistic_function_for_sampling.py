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
"""A function to compute a statistic from :class:`.Sampling`."""

from __future__ import annotations

from gemseo_umdo.formulations.functions.statistic_function import StatisticFunction


class StatisticFunctionForSampling(StatisticFunction):
    """A function to compute a statistic from :class:`.Sampling`."""

    @property
    def _observable_name(self) -> str:
        """The name of the observable."""
        return f"{self._estimate_statistic.__class__.__name__}[{self._function_name}]"
