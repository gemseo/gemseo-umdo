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

from operator import ge
from operator import le

import pytest

from gemseo_umdo.statistic.settings.probability import Probability_Settings


@pytest.mark.parametrize(
    ("settings", "expression"),
    [
        (Probability_Settings(), "P[f >= 0.0]"),
        (Probability_Settings(threshold=0.0), "P[f >= 0.0]"),
        (Probability_Settings(threshold=1.0, greater=False), "P[f <= 1.0]"),
    ],
    ids=["default", "explicit-default-threshold", "lower"],
)
def test_compute_expression(settings, expression):
    """Check that the name of a probability always includes its threshold."""
    assert settings.compute_expression("f") == expression


@pytest.mark.parametrize(
    ("greater", "operator"), [(True, ge), (False, le)], ids=["greater", "lower"]
)
def test_comparison_operator(greater, operator):
    """Check the operator comparing samples to the threshold."""
    settings = Probability_Settings(greater=greater)
    assert settings.comparison_operator is operator
    assert "comparison_operator" not in settings.model_dump()
