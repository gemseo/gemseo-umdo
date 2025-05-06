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
"""Defining the bars of the truss problem."""

from __future__ import annotations

from typing import NamedTuple

from gemseo_umdo.use_cases.truss.bar import Bar


class Bars(NamedTuple):
    """The bars of the truss model."""

    # Oblical bars
    bar_0: Bar = Bar(area=1e-3)
    bar_1: Bar = Bar(area=1e-3)
    bar_2: Bar = Bar(area=1e-3)
    bar_3: Bar = Bar(area=1e-3)
    bar_4: Bar = Bar(area=1e-3)
    bar_5: Bar = Bar(area=1e-3)
    bar_6: Bar = Bar(area=1e-3)
    bar_7: Bar = Bar(area=1e-3)
    bar_8: Bar = Bar(area=1e-3)
    bar_9: Bar = Bar(area=1e-3)
    bar_10: Bar = Bar(area=1e-3)
    bar_11: Bar = Bar(area=1e-3)
    # Bottom horizontal bars
    bar_12: Bar = Bar()
    bar_13: Bar = Bar()
    bar_14: Bar = Bar()
    bar_15: Bar = Bar()
    bar_16: Bar = Bar()
    bar_17: Bar = Bar()
    # Top horizontal bars
    bar_18: Bar = Bar()
    bar_19: Bar = Bar()
    bar_20: Bar = Bar()
    bar_21: Bar = Bar()
    bar_22: Bar = Bar()
