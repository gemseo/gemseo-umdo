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
"""Tests for the settings of the margin."""

from __future__ import annotations

import pytest
from gemseo.util.testing.helper import assert_exception

from gemseo_umdo.statistic.settings.margin import Margin_Settings


@pytest.mark.parametrize("factor", [0, -0.0], ids=["zero", "negative-zero"])
def test_zero_factor(factor, snapshot):
    """Check that a zero factor raises an error at instantiation."""
    with assert_exception(ValueError, snapshot):
        Margin_Settings(factor=factor)


@pytest.mark.parametrize(
    ("factor", "from_below", "expected"),
    [
        (3.0, False, 3.0),
        (-3.0, False, 3.0),
        (3.0, True, -3.0),
        (-3.0, True, -3.0),
    ],
    ids=["above-positive", "above-negative", "below-positive", "below-negative"],
)
def test_make_conservative(factor, from_below, expected):
    """Check that the factor of a conservative margin has the right sign."""
    settings = Margin_Settings(factor=factor)
    conservative_settings = settings.make_conservative(from_below)
    assert conservative_settings.factor == expected
    assert settings.factor == factor
