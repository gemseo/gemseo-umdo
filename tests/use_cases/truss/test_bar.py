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

from dataclasses import fields

from gemseo_umdo.use_cases.truss.bar import Bar


def test_default():
    """Check Bar with default values."""
    bar = Bar()
    assert bar.area == 2e-3
    assert bar.young_modulus == 2.1e11
    assert [f.name for f in fields(bar)] == ["area", "young_modulus"]


def test_custom():
    """Check Bar with custom values."""
    bar = Bar(area=1, young_modulus=2)
    assert bar.area == 1
    assert bar.young_modulus == 2
