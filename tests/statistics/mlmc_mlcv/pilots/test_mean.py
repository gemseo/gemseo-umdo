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
"""Tests for the mean-based pilot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import nan
from numpy.testing import assert_almost_equal

from gemseo_umdo.statistics.multilevel.mlmc_mlcv.mlmc_mlcv import MLMCMLCV
from gemseo_umdo.statistics.multilevel.mlmc_mlcv.pilots.mean import Mean

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture()
def pilot() -> Mean:
    """The mean-based pilot."""
    return Mean(array([2.0, 2.0, 2.0]), array([1, 2, 3]))


@pytest.fixture(scope="module")
def parameters() -> tuple[NDArray[float], NDArray[float]]:
    """The parameters of the pilot."""
    return (array([1.0, 2.0, 3.0]), array([0.1, 0.2]))


def test_compute_statistic(pilot, samples, parameters):
    """Check the computation of the statistic."""
    _, statistic = pilot.compute_next_level_and_statistic(
        [1], array([30, 20, 10]), samples, *parameters, MLMCMLCV.Variant.MLMC_MLCV
    )
    assert_almost_equal(statistic, -0.3)

    _, statistic = pilot.compute_next_level_and_statistic(
        [2], array([30, 20, 10]), samples, *parameters, MLMCMLCV.Variant.MLMC_MLCV
    )
    assert_almost_equal(statistic, -0.7)


@pytest.mark.parametrize(
    ("variant", "expected_V_l", "expected_x"),
    [
        (MLMCMLCV.Variant.MLMC_MLCV, 0.0, [-0.3, -0.3, -0.3]),
        (MLMCMLCV.Variant.MLMC_CV_0, 0.0022222, [-0.3, -0.4, -0.4]),
    ],
)
def test_compute_V_l_delta(  # noqa: N802
    pilot,
    samples,
    parameters,
    variant,
    expected_V_l,  # noqa: N803
    expected_x,  # noqa: N803
):
    """Check the computation of the V_l and delta."""
    V_l = pilot._compute_V_l([1], samples, *parameters, variant)  # noqa: N806
    delta = pilot._Mean__delta
    assert_almost_equal(V_l, array([nan, expected_V_l, nan]))
    assert len(delta) == 3
    for x, y in zip(delta, [array([]), array(expected_x), array([])]):
        assert_almost_equal(x, y)
