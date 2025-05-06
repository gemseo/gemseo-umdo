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
from __future__ import annotations

from gemseo_umdo.use_cases.truss.bar import Bar
from gemseo_umdo.use_cases.truss.bars import Bars


def test_default():
    """Check Bars with default values."""
    bars = Bars()
    assert bars._fields == tuple(f"bar_{i}" for i in range(23))
    horizontal_bar = Bar()
    oblical_bar = Bar(area=1e-3)
    for i in range(12):
        assert getattr(bars, f"bar_{i}") == oblical_bar

    for i in range(12, 23):
        assert getattr(bars, f"bar_{i}") == horizontal_bar


def test_custom():
    """Check Bars with custom values."""
    bar = Bar(area=1)
    bars = Bars(**{f"bar_{i}": bar for i in range(23)})
    for i in range(23):
        assert getattr(bars, f"bar_{i}") == bar
