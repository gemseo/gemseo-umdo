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
from typing import Callable

import pytest

if TYPE_CHECKING:
    from numpy import ndarray


@pytest.fixture
def model() -> Callable[[ndarray], ndarray]:
    """The model f."""

    def f(x: ndarray) -> ndarray:
        r"""The model f_{\ell}.

        Args:
            x: The input data.

        Returns:
            The output data.
        """
        return 2 * x

    return f
