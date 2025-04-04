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

from typing import TYPE_CHECKING

import pytest
from numpy import array

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture
def samples() -> list[NDArray[float]]:
    """The samples used to test MLMC-MLCV methods."""
    return [
        array([
            [1.0, 1.2, 1.0, 2.0, 3.0],
            [2.0, 2.3, -3.0, -2.0, -1.0],
            [3.0, 3.4, 2.0, 3.0, 1.0],
        ]),
        array([[1.1, 1.4, 0.1, 0.2], [2.1, 2.5, -0.2, -0.3], [3.1, 3.5, 0.3, 0.2]]),
        array([[1.2, 1.6, 0.1, 0.2], [2.2, 2.7, -0.2, -0.2], [3.2, 3.7, 0.3, 0.2]]),
    ]
