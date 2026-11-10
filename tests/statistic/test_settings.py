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
"""Tests for the settings of the statistics."""

from __future__ import annotations

import pytest
from gemseo.util.testing.helper import assert_exception

from gemseo_umdo.statistic import Margin_Settings
from gemseo_umdo.statistic import Mean_Settings
from gemseo_umdo.statistic import Probability_Settings
from gemseo_umdo.statistic import Quantile_Settings
from gemseo_umdo.statistic import StandardDeviation_Settings
from gemseo_umdo.statistic import Variance_Settings
from gemseo_umdo.statistic.settings import BaseStatisticSettings


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
)
def test_defaults(settings_class, name, fields):
    """Check the target class name and the default values of the fields."""
    settings = settings_class()
    assert settings.target_class_name == name
    assert settings.model_dump() == fields


@pytest.mark.parametrize(
    ("settings_class", "kwargs"),
    [
        (Margin_Settings, {"factor": 0}),
        (Margin_Settings, {"factor": -0.0}),
        (Quantile_Settings, {"level": 0.0}),
        (Quantile_Settings, {"level": 1.0}),
        (Quantile_Settings, {"method": "foo"}),
    ],
    ids=["zero-factor", "negative-zero-factor", "level-zero", "level-one", "method"],
)
def test_invalid_settings(settings_class, kwargs, snapshot):
    """Check that invalid settings raise an error at instantiation."""
    with assert_exception(ValueError, snapshot):
        settings_class(**kwargs)


@pytest.mark.parametrize(
    ("factor", "from_below", "expected"),
    [
        (3.0, False, 3.0),
        (-3.0, False, 3.0),
        (3.0, True, -3.0),
        (-3.0, True, -3.0),
    ],
)
def test_margin_make_conservative(factor, from_below, expected):
    """Check that the factor of a conservative margin has the right sign."""
    settings = Margin_Settings(factor=factor)
    conservative_settings = settings.make_conservative(from_below)
    assert conservative_settings.factor == expected
    assert settings.factor == factor


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
def test_quantile_make_conservative(level, from_below, expected):
    """Check that the level of a conservative quantile is on the right side."""
    settings = Quantile_Settings(level=level, method="hazen")
    conservative_settings = settings.make_conservative(from_below)
    assert conservative_settings.level == expected
    assert conservative_settings.method == "hazen"
    assert settings.level == level


@pytest.mark.parametrize("from_below", [False, True])
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
        (Probability_Settings(), {}),
        (Probability_Settings(greater=False), {"greater": False}),
        (Margin_Settings().make_conservative(True), {"factor": -2.0}),
    ],
)
def test_get_name_parameters(settings, parameters):
    """Check the parameters used in the name of a statistic."""
    assert settings.get_name_parameters() == parameters


def test_from_name():
    """Check the default settings created from the name of a statistic."""
    assert BaseStatisticSettings.from_name("Margin") == Margin_Settings()


def test_from_name_unknown():
    """Check the error raised when no settings class is defined for a statistic.

    The message is not snapshotted
    as it lists the settings classes defined by the other tests, if any.
    """
    prefix = "The statistic 'Unknown' has no settings class; available statistics: "
    with pytest.raises(ValueError, match=r"^The statistic 'Unknown'") as error:
        BaseStatisticSettings.from_name("Unknown")

    message = str(error.value)
    assert message.startswith(prefix)
    assert message.endswith(".")
    names = message.removeprefix(prefix).removesuffix(".").split(", ")
    assert names == sorted(names)
    assert {
        "Margin",
        "Mean",
        "Probability",
        "Quantile",
        "StandardDeviation",
        "Variance",
    } <= set(names)


def test_from_name_with_subclass():
    """Check that a subclass of a settings class does not shadow it."""

    class OtherMargin_Settings(Margin_Settings):  # noqa: N801
        """Another margin."""

    assert OtherMargin_Settings().target_class_name == "OtherMargin"
    assert type(BaseStatisticSettings.from_name("Margin")) is Margin_Settings
    assert type(BaseStatisticSettings.from_name("OtherMargin")) is OtherMargin_Settings


def test_from_name_with_intermediate_base_class():
    """Check that a statistic deriving from an intermediate base class is found."""

    class BaseFooStatisticSettings(BaseStatisticSettings):
        """An intermediate base class, bound to no statistic."""

    class IntermediateFoo_Settings(BaseFooStatisticSettings):  # noqa: N801
        """The settings of a statistic deriving from an intermediate base class."""

    assert not BaseFooStatisticSettings().target_class_name
    assert (
        type(BaseStatisticSettings.from_name("IntermediateFoo"))
        is IntermediateFoo_Settings
    )
