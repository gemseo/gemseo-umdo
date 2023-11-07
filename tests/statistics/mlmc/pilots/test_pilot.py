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
"""Tests for the base class Pilot."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterable
from typing import Sequence

import pytest
from gemseo.utils.testing.helpers import concretize_classes
from numpy import array
from numpy import ndarray
from numpy.testing import assert_equal

from gemseo_umdo.statistics.multilevel.pilot import Pilot

if TYPE_CHECKING:
    from numpy._typing import NDArray


class MyPilot(Pilot):
    """A dummy pilot."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(array([2.0, 2.0, 2.0]), array([1, 2, 3]))

    def _compute_V_l(  # noqa: D107  N802
        self,
        levels: Iterable[int],
        samples: Sequence[NDArray[float]],
        a: int,
        b: int,
    ) -> ndarray:
        return (
            array([(data[:, 1] - data[:, 0]).mean() for data in samples])
            + array([1.0, 2.0, 3.0])
            + a
            + b
        )


@pytest.fixture(scope="module")
def pilot() -> MyPilot:
    """The pilot and the samples."""
    with concretize_classes(MyPilot):
        return MyPilot()


def test_compute_V_l(pilot, samples):  # noqa: N802
    """Check the computation of the variances."""
    assert_equal(
        pilot._compute_V_l([1], samples, 1, 2),
        array([4.25, 5.35, 6.45]),
    )


def test_compute_next_level(pilot, samples):
    """Check the computation of the next level."""
    next_level, statistic = pilot.compute_next_level_and_statistic(
        [1], array([12, 43, 19]), samples, 1, 2
    )
    assert next_level == 0
