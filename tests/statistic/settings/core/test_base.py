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

from gemseo_umdo.statistic.settings.core.base import BaseStatisticSettings
from gemseo_umdo.statistic.settings.margin import Margin_Settings
from gemseo_umdo.statistic.settings.mean import Mean_Settings
from gemseo_umdo.statistic.settings.probability import Probability_Settings
from gemseo_umdo.statistic.settings.quantile import Quantile_Settings
from gemseo_umdo.statistic.settings.standard_deviation import StandardDeviation_Settings
from gemseo_umdo.statistic.settings.variance import Variance_Settings


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


@pytest.mark.parametrize(
    "name",
    ["Margin", "margin", "MARGIN", "marGin"],
    ids=["exact", "lower", "upper", "mixed"],
)
def test_from_name(name):
    """Check the default settings created from a case-insensitive name."""
    assert BaseStatisticSettings.from_name(name) == Margin_Settings()


@pytest.mark.usefixtures("isolated_subclasses")
def test_from_name_with_names_equal_up_to_case(snapshot):
    """Check from_name with statistics whose names differ only by the case."""

    class Foo_Settings(BaseStatisticSettings):  # noqa: N801
        """The settings of a statistic."""

    class FOO_Settings(BaseStatisticSettings):  # noqa: N801
        """The settings of another statistic."""

    assert type(BaseStatisticSettings.from_name("Foo")) is Foo_Settings
    assert type(BaseStatisticSettings.from_name("FOO")) is FOO_Settings
    with assert_exception(ValueError, snapshot):
        BaseStatisticSettings.from_name("foo")


@pytest.mark.usefixtures("isolated_subclasses")
def test_from_name_unknown(snapshot):
    """Check the error raised when no settings class is defined for a statistic."""
    with assert_exception(ValueError, snapshot):
        BaseStatisticSettings.from_name("Unknown")


def _define_subclass_with_suffix() -> tuple[
    dict[type[BaseStatisticSettings], str], dict[str, type[BaseStatisticSettings]]
]:
    """Define a subclass of a settings class whose name has the suffix.

    Returns:
        The target class names of the defined classes
        and the settings classes expected from their names.
    """

    class OtherMargin_Settings(Margin_Settings):  # noqa: N801
        """Another margin."""

    return (
        {OtherMargin_Settings: "OtherMargin"},
        {"OtherMargin": OtherMargin_Settings, "Margin": Margin_Settings},
    )


def _define_subclass_without_suffix() -> tuple[
    dict[type[BaseStatisticSettings], str], dict[str, type[BaseStatisticSettings]]
]:
    """Define a subclass of a settings class whose name has not the suffix.

    GEMSEO binds it to the statistic of its parent class,
    but it does not replace this parent class.

    Returns:
        The target class names of the defined classes
        and the settings classes expected from their names.
    """

    class MyMargin(Margin_Settings):
        """A margin."""

    return {MyMargin: "Margin"}, {"Margin": Margin_Settings}


def _define_intermediate_base_class() -> tuple[
    dict[type[BaseStatisticSettings], str], dict[str, type[BaseStatisticSettings]]
]:
    """Define a statistic deriving from an intermediate base class.

    Returns:
        The target class names of the defined classes
        and the settings classes expected from their names.
    """

    class BaseFooStatisticSettings(BaseStatisticSettings):
        """An intermediate base class, bound to no statistic."""

    class IntermediateFoo_Settings(BaseFooStatisticSettings):  # noqa: N801
        """The settings of a statistic deriving from an intermediate base class."""

    return (
        {BaseFooStatisticSettings: "", IntermediateFoo_Settings: "IntermediateFoo"},
        {"IntermediateFoo": IntermediateFoo_Settings},
    )


@pytest.mark.usefixtures("isolated_subclasses")
@pytest.mark.parametrize(
    "define_classes",
    [
        _define_subclass_with_suffix,
        _define_subclass_without_suffix,
        _define_intermediate_base_class,
    ],
    ids=["subclass-with-suffix", "subclass-without-suffix", "intermediate-base"],
)
def test_from_name_registration(define_classes):
    """Check that a class is registered under the target class name of GEMSEO.

    GEMSEO sets this name only for a class whose name has the suffix `_Settings`,
    so the other classes are not registered.
    """
    classes_to_target_class_names, names_to_classes = define_classes()
    for settings_class, name in classes_to_target_class_names.items():
        assert settings_class().target_class_name == name

    for name, settings_class in names_to_classes.items():
        assert type(BaseStatisticSettings.from_name(name)) is settings_class


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
