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
"""Tests for the settings of the quantile."""

from __future__ import annotations

import pytest
from gemseo.util.testing.helper import assert_exception

from gemseo_umdo.statistic.quantile_settings import Quantile_Settings


@pytest.mark.parametrize(
    "kwargs",
    [{"level": 0.0}, {"level": 1.0}, {"method": "foo"}],
    ids=["level-zero", "level-one", "method"],
)
def test_invalid_settings(kwargs, snapshot):
    """Check that invalid settings raise an error at instantiation."""
    with assert_exception(ValueError, snapshot):
        Quantile_Settings(**kwargs)


@pytest.mark.parametrize(
    ("level", "from_below", "expected"),
    [
        (0.9, False, 0.9),
        (0.1, False, 0.9),
        (0.9, True, 0.1),
        (0.1, True, 0.1),
        (0.07, False, 0.93),
    ],
)
def test_make_conservative(level, from_below, expected):
    """Check that the level of a conservative quantile is on the right side."""
    settings = Quantile_Settings(level=level, method="hazen")
    conservative_settings = settings.make_conservative(from_below)
    assert conservative_settings.level == expected
    assert conservative_settings.method == "hazen"
    assert settings.level == level
