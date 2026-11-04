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

from typing import TYPE_CHECKING

import pytest

from gemseo_umdo.disciplines.noiser_factory import NoiserFactory
from gemseo_umdo.formulations._statistics.control_variate.factory import (
    ControlVariateEstimatorFactory,
)
from gemseo_umdo.formulations._statistics.iterative_sampling.factory import (
    SamplingEstimatorFactory as IterativeSamplingEstimatorFactory,
)
from gemseo_umdo.formulations._statistics.pce.factory import PCEEstimatorFactory
from gemseo_umdo.formulations._statistics.sampling.factory import (
    SamplingEstimatorFactory,
)
from gemseo_umdo.formulations._statistics.taylor_polynomial.factory import (
    TaylorPolynomialEstimatorFactory,
)
from gemseo_umdo.formulations.factory import UMDOFormulationsFactory
from gemseo_umdo.statistics.multilevel.mlmc.pilots.factory import MLMCPilotFactory
from gemseo_umdo.statistics.multilevel.mlmc_mlcv.pilots.factory import (
    MLMCMLCVPilotFactory,
)

if TYPE_CHECKING:
    from gemseo.core.base_factory import BaseFactory


@pytest.mark.parametrize(
    "factory_class",
    [
        pytest.param(NoiserFactory, id="NoiserFactory"),
        pytest.param(UMDOFormulationsFactory, id="UMDOFormulationsFactory"),
        pytest.param(
            ControlVariateEstimatorFactory, id="ControlVariateEstimatorFactory"
        ),
        pytest.param(
            IterativeSamplingEstimatorFactory,
            id="iterative_sampling.SamplingEstimatorFactory",
        ),
        pytest.param(PCEEstimatorFactory, id="PCEEstimatorFactory"),
        pytest.param(SamplingEstimatorFactory, id="sampling.SamplingEstimatorFactory"),
        pytest.param(
            TaylorPolynomialEstimatorFactory, id="TaylorPolynomialEstimatorFactory"
        ),
        pytest.param(MLMCPilotFactory, id="MLMCPilotFactory"),
        pytest.param(MLMCMLCVPilotFactory, id="MLMCMLCVPilotFactory"),
    ],
)
def test_factory(reset_factory, factory_class: type[BaseFactory]) -> None:
    """Check that a factory imports its packages without error and finds classes."""
    factory = factory_class()
    assert factory.failed_imports == {}
    assert factory.class_names
