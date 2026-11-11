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

import gc

import pytest
from gemseo.util.testing.helper import assert_exception

from gemseo_umdo.statistic.core.base_settings import BaseStatisticSettings
from gemseo_umdo.statistic.margin_settings import Margin_Settings
from gemseo_umdo.statistic.mean_settings import Mean_Settings
from gemseo_umdo.statistic.probability_settings import Probability_Settings
from gemseo_umdo.statistic.quantile_settings import Quantile_Settings
from gemseo_umdo.statistic.standard_deviation_settings import StandardDeviation_Settings
from gemseo_umdo.statistic.variance_settings import Variance_Settings


def _get_subclasses(cls: type) -> set[type]:
    """Return the subclasses of a class, recursively.

    Args:
        cls: The class.

    Returns:
        The subclasses of the class.
    """
    subclasses = set()
    for subclass in cls.__subclasses__():
        subclasses |= {subclass, *_get_subclasses(subclass)}

    return subclasses


@pytest.fixture
def isolated_subclasses():
    """Check that the settings classes defined by a test do not outlive it."""
    # The classes defined in a test are freed with the frame of the test,
    # but their reference cycles require a garbage collection,
    # including those of the previous tests.
    gc.collect()
    subclasses = _get_subclasses(BaseStatisticSettings)
    yield
    gc.collect()
    assert _get_subclasses(BaseStatisticSettings) == subclasses


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


def test_from_name():
    """Check the default settings created from the name of a statistic."""
    assert BaseStatisticSettings.from_name("Margin") == Margin_Settings()


@pytest.mark.usefixtures("isolated_subclasses")
def test_from_name_unknown(snapshot):
    """Check the error raised when no settings class is defined for a statistic."""
    with assert_exception(ValueError, snapshot):
        BaseStatisticSettings.from_name("Unknown")


@pytest.mark.usefixtures("isolated_subclasses")
def test_from_name_with_subclass():
    """Check that a subclass of a settings class does not shadow it."""

    class OtherMargin_Settings(Margin_Settings):  # noqa: N801
        """Another margin."""

    assert OtherMargin_Settings().target_class_name == "OtherMargin"
    assert type(BaseStatisticSettings.from_name("Margin")) is Margin_Settings
    assert type(BaseStatisticSettings.from_name("OtherMargin")) is OtherMargin_Settings


@pytest.mark.usefixtures("isolated_subclasses")
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


@pytest.mark.usefixtures("isolated_subclasses")
def test_from_name_with_post_init_reading_required_field():
    """Check that from_name does not instantiate the settings classes.

    A settings class whose post-initialization reads a required field
    cannot be instantiated without this field.
    """

    class Doubled_Settings(BaseStatisticSettings):  # noqa: N801
        """The settings of a statistic with a required field."""

        value: float

        def model_post_init(self, context, /) -> None:
            self.value *= 2

    assert BaseStatisticSettings.from_name("Mean") == Mean_Settings()


@pytest.mark.usefixtures("isolated_subclasses")
def test_from_name_with_redefinition():
    """Check that a redefined settings class replaces the previous one.

    This is the case of a notebook cell defining a class and run again,
    while the previous class is still alive.
    """

    class Foo_Settings(BaseStatisticSettings):  # noqa: N801
        """The settings of a statistic."""

        value: float = 1.0

    previous_class = Foo_Settings

    class Foo_Settings(BaseStatisticSettings):  # noqa: N801
        """The settings of the same statistic, redefined."""

        value: float = 2.0

    assert previous_class is not Foo_Settings
    assert BaseStatisticSettings.from_name("Foo") == Foo_Settings()


@pytest.mark.usefixtures("isolated_subclasses")
def test_from_name_with_subclass_without_suffix():
    """Check that a subclass whose name has not the suffix is not a statistic.

    GEMSEO binds it to the statistic of its parent class,
    but it does not replace this parent class.
    """

    class MyMargin(Margin_Settings):
        """A margin."""

    assert MyMargin().target_class_name == "Margin"
    assert type(BaseStatisticSettings.from_name("Margin")) is Margin_Settings
