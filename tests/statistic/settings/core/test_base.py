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
"""Tests for the base settings of the statistics."""

from __future__ import annotations

import pytest

from gemseo_umdo.statistic.settings.margin import Margin_Settings
from gemseo_umdo.statistic.settings.mean import Mean_Settings
from gemseo_umdo.statistic.settings.probability import Probability_Settings
from gemseo_umdo.statistic.settings.quantile import Quantile_Settings
from gemseo_umdo.statistic.settings.standard_deviation import StandardDeviation_Settings
from gemseo_umdo.statistic.settings.variance import Variance_Settings


@pytest.mark.parametrize(
    ("settings_class", "name", "fields"),
    [
        (Mean_Settings, "Mean", {}),
        (Variance_Settings, "Variance", {}),
        (StandardDeviation_Settings, "StandardDeviation", {}),
        (Margin_Settings, "Margin", {"factor": 2.0}),
        (Probability_Settings, "Probability", {"threshold": 0.0, "greater": True}),
        (Quantile_Settings, "Quantile", {"level": 0.95, "method": "linear"}),
    ],
    ids=[
        "mean",
        "variance",
        "standard-deviation",
        "margin",
        "probability",
        "quantile",
    ],
)
def test_defaults(settings_class, name, fields):
    """Check the target class name and the default values of the fields."""
    settings = settings_class()
    assert settings.target_class_name == name
    assert settings.model_dump() == fields


@pytest.mark.parametrize("from_below", [False, True], ids=["above", "below"])
def test_default_make_conservative(from_below):
    """Check that the settings are unchanged by default."""
    settings = Probability_Settings(threshold=1.0)
    assert settings.make_conservative(from_below) is settings


@pytest.mark.parametrize(
    ("settings", "parameters"),
    [
        (Mean_Settings(), {}),
        (Margin_Settings(), {"factor": 2.0}),
        (Quantile_Settings(), {"level": 0.95}),
        (Quantile_Settings(method="hazen"), {"level": 0.95, "method": "hazen"}),
        (Probability_Settings(), {"threshold": 0.0}),
        (Probability_Settings(greater=False), {"threshold": 0.0, "greater": False}),
        (Margin_Settings().make_conservative(True), {"factor": -2.0}),
    ],
    ids=[
        "mean",
        "margin",
        "quantile",
        "quantile-hazen",
        "probability",
        "probability-lower",
        "margin-conservative",
    ],
)
def test_get_name_parameters(settings, parameters):
    """Check the parameters used in the name of a statistic."""
    assert settings.get_name_parameters() == parameters
