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

from gemseo_umdo.use_cases.truss.forces import Forces


def test_default():
    """Check Forces with default values."""
    forces = Forces()
    assert forces._fields == tuple(f"force_{i}" for i in range(7, 13))
    for i in range(7, 13):
        assert getattr(forces, f"force_{i}") == 5e4


def test_custom():
    """Check Forces with custom values."""
    forces = Forces(**{f"force_{i}": i for i in range(7, 13)})
    for i in range(7, 13):
        assert getattr(forces, f"force_{i}") == i
