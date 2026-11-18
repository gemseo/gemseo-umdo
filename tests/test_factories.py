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
"""Tests for the factories of gemseo-umdo."""

from __future__ import annotations

import pytest
from gemseo.core.base_factory import BaseFactory

from gemseo_umdo.discipline.noiser.factory import NoiserFactory
from gemseo_umdo.formulation._estimator.control_variate.factory import (
    ControlVariateEstimatorFactory,
)
from gemseo_umdo.formulation._estimator.iterative_sampling.factory import (
    IterativeSamplingEstimatorFactory,
)
from gemseo_umdo.formulation._estimator.pce.factory import PCEEstimatorFactory
from gemseo_umdo.formulation._estimator.sampling.factory import SamplingEstimatorFactory
from gemseo_umdo.formulation._estimator.taylor_polynomial.factory import (
    TaylorPolynomialEstimatorFactory,
)
from gemseo_umdo.formulation.factory import UMDOFormulationFactory
from gemseo_umdo.statistic.multilevel.mlmc.pilot.factory import MLMCPilotFactory
from gemseo_umdo.statistic.multilevel.mlmc_mlcv.pilot.factory import (
    MLMCMLCVPilotFactory,
)
from gemseo_umdo.statistic.settings.core.factory import StatisticSettingsFactory


@pytest.mark.parametrize(
    "factory_class",
    [
        pytest.param(NoiserFactory, id="NoiserFactory"),
        pytest.param(UMDOFormulationFactory, id="UMDOFormulationFactory"),
        pytest.param(
            ControlVariateEstimatorFactory, id="ControlVariateEstimatorFactory"
        ),
        pytest.param(
            IterativeSamplingEstimatorFactory,
            id="IterativeSamplingEstimatorFactory",
        ),
        pytest.param(PCEEstimatorFactory, id="PCEEstimatorFactory"),
        pytest.param(SamplingEstimatorFactory, id="SamplingEstimatorFactory"),
        pytest.param(
            TaylorPolynomialEstimatorFactory, id="TaylorPolynomialEstimatorFactory"
        ),
        pytest.param(MLMCPilotFactory, id="MLMCPilotFactory"),
        pytest.param(MLMCMLCVPilotFactory, id="MLMCMLCVPilotFactory"),
        pytest.param(StatisticSettingsFactory, id="StatisticSettingsFactory"),
    ],
)
def test_factory(reset_factory, factory_class: type[BaseFactory]) -> None:
    """Check that a factory imports its packages without error and finds classes."""
    factory = factory_class()
    assert factory.failed_imports == {}
    assert factory.class_names


def test_iterative_sampling_factory_class_names(reset_factory) -> None:
    """Check that the iterative factory lists only the concrete estimators."""
    factory = IterativeSamplingEstimatorFactory()
    assert factory.class_names == [
        "Margin",
        "Mean",
        "Probability",
        "StandardDeviation",
        "Variance",
    ]


def test_iterative_sampling_factory_keeps_base_cache(reset_factory) -> None:
    """Check that the iterative factory does not alter the cache of BaseFactory."""
    factory = IterativeSamplingEstimatorFactory()
    factory.class_names  # noqa: B018
    base_name_to_class_info = BaseFactory._name_to_class_info.fget(factory)
    assert "BaseIterativeSamplingEstimator" in base_name_to_class_info


def test_statistic_settings_factory_class_names(reset_factory) -> None:
    """Check that the factory of settings of statistics finds the statistics."""
    assert StatisticSettingsFactory().class_names == [
        "BaseStatisticSettings",
        "Margin_Settings",
        "Mean_Settings",
        "Probability_Settings",
        "Quantile_Settings",
        "StandardDeviation_Settings",
        "Variance_Settings",
    ]
