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
"""Tests for the settings of the probability."""

from __future__ import annotations

import pytest

from gemseo_umdo.statistic.probability_settings import Probability_Settings


@pytest.mark.parametrize(
    ("settings", "expression"),
    [
        (Probability_Settings(), "P[f >= 0.0]"),
        (Probability_Settings(threshold=0.0), "P[f >= 0.0]"),
        (Probability_Settings(threshold=1.0, greater=False), "P[f <= 1.0]"),
    ],
)
def test_compute_expression(settings, expression):
    """Check that the name of a probability always includes its threshold."""
    assert settings.compute_expression("f") == expression
