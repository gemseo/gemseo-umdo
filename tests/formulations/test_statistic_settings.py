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

import sys
from textwrap import dedent

import pytest
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.formulation import MDF_Settings
from gemseo.util.testing.helper import assert_exception
from numpy import array
from numpy.testing import assert_almost_equal

import gemseo_umdo.formulations
import gemseo_umdo.scenarios
from gemseo_umdo.formulations import BaseStatisticSettings
from gemseo_umdo.formulations import Margin_Settings
from gemseo_umdo.formulations import Mean_Settings
from gemseo_umdo.formulations import Probability_Settings
from gemseo_umdo.formulations import Quantile_Settings
from gemseo_umdo.formulations import Sampling_Settings
from gemseo_umdo.formulations import StandardDeviation_Settings
from gemseo_umdo.formulations import TaylorPolynomial_Settings
from gemseo_umdo.formulations import Variance_Settings
from gemseo_umdo.scenarios import UMDOScenario


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
    with pytest.raises(
        ValueError,
        match=(
            r"^The statistic 'Unknown' has no settings class; available statistics: "
            r".*Margin, Mean, .*Probability, Quantile, StandardDeviation, Variance.*\.$"
        ),
    ):
        BaseStatisticSettings.from_name("Unknown")


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


def test_unsupported_statistic(disciplines, design_space, uncertain_space, snapshot):
    """Check the error raised for a statistic not supported by a formulation."""
    scenario = UMDOScenario(
        disciplines,
        design_space,
        uncertain_space,
        statistic_estimation_settings=TaylorPolynomial_Settings(),
        formulation_settings=MDF_Settings(),
    )
    with assert_exception(ValueError, snapshot):
        scenario.add_objective("f", Quantile_Settings())


@pytest.mark.parametrize(
    ("package", "names"),
    [
        (
            gemseo_umdo.formulations,
            {
                "BaseStatisticSettings",
                "ControlVariate_Settings",
                "DOEAlgorithmName",
                "Margin_Settings",
                "Mean_Settings",
                "PCE_Settings",
                "Probability_Settings",
                "Quantile_Settings",
                "Sampling_Settings",
                "SequentialSampling_Settings",
                "StandardDeviation_Settings",
                "Surrogate_Settings",
                "TaylorPolynomial_Settings",
                "Variance_Settings",
            },
        ),
        (gemseo_umdo.scenarios, {"UEvaluationScenario", "UMDOScenario"}),
    ],
)
def test_public_names(package, names):
    """Check the names exported by the public packages."""
    assert set(package.__all__) == names
    assert all(hasattr(package, name) for name in names)


_user_statistic_module = '''
from __future__ import annotations

from gemseo_umdo.formulations import BaseStatisticSettings
from gemseo_umdo.formulations._statistics.sampling.base_sampling_estimator import (
    BaseSamplingEstimator,
)


class ScaledMean_Settings(BaseStatisticSettings):
    """The settings of a scaled mean."""

    scale: float = 1.0


class ScaledMean(BaseSamplingEstimator):
    """Estimator of a scaled mean."""

    def __init__(self, settings):
        super().__init__(settings)

    def estimate_statistic(self, samples):
        return self._settings.scale * samples.mean(0)

    def compute_jacobian(self, samples, jac_samples):
        return self._settings.scale * jac_samples.mean(0)
'''
"""A module defining a statistic of the user."""


@pytest.mark.usefixtures("reset_factory")
def test_user_statistic(
    disciplines, design_space, uncertain_space, tmp_path, monkeypatch
):
    """Check that a user can add a statistic with its settings and estimator."""
    module_name = "user_statistic_for_gemseo_umdo"
    (tmp_path / f"{module_name}.py").write_text(dedent(_user_statistic_module))
    monkeypatch.setenv("GEMSEO_PATH", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    user_module = __import__(module_name)

    def execute(statistic_settings):
        scenario = UMDOScenario(
            disciplines,
            design_space,
            uncertain_space,
            statistic_estimation_settings=Sampling_Settings(
                n_samples=10, estimate_statistics_iteratively=False
            ),
            formulation_settings=MDF_Settings(),
        )
        scenario.add_objective("f", statistic_settings)
        scenario.execute(CustomDOE_Settings(samples=array([[1.0, 1.0, 1.0]])))
        return scenario.formulation.problem

    problem = execute(user_module.ScaledMean_Settings(scale=3.0))
    reference_problem = execute("Mean")
    assert problem.objective.name == "scaled_mean[f; 3.0]"
    assert_almost_equal(
        problem.database.last_item[problem.objective.name],
        3.0 * reference_problem.database.last_item["E[f]"],
    )
