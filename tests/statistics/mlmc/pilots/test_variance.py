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
"""Tests for the variance-based pilot."""
from __future__ import annotations

import pytest
from gemseo_umdo.statistics.multilevel.mlmc.pilots.variance import Variance
from numpy import array
from numpy import nan
from numpy.testing import assert_almost_equal


@pytest.fixture
def pilot() -> Variance:
    """The variance-based pilot."""
    return Variance(array([2.0, 2.0, 2.0]), array([1, 2, 3]))


def test_compute_statistics(pilot, samples):
    """Check the computation of the statistic."""
    _, statistic = pilot.compute_next_level_and_statistic(
        [1], array([30, 20, 10]), samples
    )
    assert_almost_equal(statistic, array([-0.0525]))

    _, statistic = pilot.compute_next_level_and_statistic(
        [2], array([30, 20, 10]), samples
    )
    assert_almost_equal(statistic, array([-0.105]))


def test_compute_V_l_delta(pilot, samples):  # noqa: N802
    """Check the computation of the V_l and delta."""
    V_l = pilot._compute_V_l([1], samples)  # noqa: N806
    delta = pilot._Variance__delta
    sigma = pilot._Variance__sigma
    assert_almost_equal(V_l, array([nan, 0.0027563, nan]))
    assert len(delta) == 3
    assert len(sigma) == 3
    for x, y in zip(delta, [array([]), array([-0.3, -0.4]), array([])]):
        assert_almost_equal(x, y)
    for x, y in zip(sigma, [array([]), array([2.5, 4.6]), array([])]):
        assert_almost_equal(x, y)
