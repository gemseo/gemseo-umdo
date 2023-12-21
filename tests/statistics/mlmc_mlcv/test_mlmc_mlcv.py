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
"""Tests for the MLMC-MLMCV algorithm."""

from __future__ import annotations

import pytest
from gemseo.algos.parameter_space import ParameterSpace

from gemseo_umdo.statistics.multilevel.mlmc_mlcv.level import Level
from gemseo_umdo.statistics.multilevel.mlmc_mlcv.mlmc_mlcv import MLMCMLCV
from gemseo_umdo.statistics.multilevel.mlmc_mlcv.pilots.factory import (
    MLMCMLCVPilotFactory,
)
from gemseo_umdo.statistics.multilevel.mlmc_mlcv.pilots.mean import Mean


@pytest.fixture(scope="module")
def levels() -> list[Level]:
    """The MLMC levels."""
    return [
        Level(lambda x: 2 * x, (lambda x: 2.1 * x, 0.8), (), 2.0),
        Level(
            lambda x: 1.8 * x,
            (lambda x: 2.1 * x, 1.04),
            (lambda x: 0.02 * x, 0.002),
            4.0,
        ),
        Level(
            lambda x: 1.8 * x,
            (lambda x: 2.1 * x, 1.04),
            (lambda x: 0.02 * x, 0.002),
            8.0,
        ),
    ]


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace:
    """The uncertain space."""
    space = ParameterSpace()
    space.add_random_variable("x", "OTUniformDistribution")
    return space


@pytest.fixture(scope="module")
def mlmc_mlcv(levels, uncertain_space):
    """The MLMC-MLCV algorithm with the default parametrization."""
    return MLMCMLCV(levels, uncertain_space, 1000.0)


@pytest.fixture(scope="module")
def samplers(mlmc_mlcv):
    """The samplers attached to the MLMC-MLCV algorithm."""
    return mlmc_mlcv._samplers


def test_pilot_factory(mlmc_mlcv):
    """Check the pilot factory."""
    assert MLMCMLCVPilotFactory == mlmc_mlcv._PILOT_FACTORY
    assert isinstance(mlmc_mlcv._MLMC__pilot_statistic_estimator, Mean)


def test_default_variant(mlmc_mlcv):
    """Check the default variant."""
    assert mlmc_mlcv._algorithm_name == "MLMC-MLCV"


def test_custom_variant(levels, uncertain_space):
    """Check a custom variant."""
    for variant in MLMCMLCV.Variant:
        mlmc_mlcv = MLMCMLCV(levels, uncertain_space, 1000.0, variant=variant)
        assert mlmc_mlcv._algorithm_name == variant.value


@pytest.mark.parametrize(
    ("level", "size", "expected_names", "variant"),
    [
        (0, 5, ["f[0]", "f[-1]", "g[0]", "g[1]", "g[2]"], MLMCMLCV.Variant.MLMC_MLCV),
        (1, 4, ["f[1]", "f[0]", "h[1]", "h[2]"], MLMCMLCV.Variant.MLMC_MLCV),
        (2, 4, ["f[2]", "f[1]", "h[1]", "h[2]"], MLMCMLCV.Variant.MLMC_MLCV),
        (0, 3, ["f[0]", "f[-1]", "g[0]"], MLMCMLCV.Variant.MLMC_CV),
        (1, 3, ["f[1]", "f[0]", "h[1]"], MLMCMLCV.Variant.MLMC_CV),
        (2, 3, ["f[2]", "f[1]", "h[2]"], MLMCMLCV.Variant.MLMC_CV),
        (0, 3, ["f[0]", "f[-1]", "g[0]"], MLMCMLCV.Variant.MLMC_CV_0),
        (1, 2, ["f[1]", "f[0]"], MLMCMLCV.Variant.MLMC_CV_0),
        (2, 2, ["f[2]", "f[1]"], MLMCMLCV.Variant.MLMC_CV_0),
        (0, 4, ["f[0]", "f[-1]", "g[0]", "g[1]"], MLMCMLCV.Variant.MLMC_MLCV_0),
        (1, 3, ["f[1]", "f[0]", "h[1]"], MLMCMLCV.Variant.MLMC_MLCV_0),
        (2, 3, ["f[2]", "f[1]", "h[1]"], MLMCMLCV.Variant.MLMC_MLCV_0),
    ],
)
def test_samplers_mlmc_cv(
    levels, uncertain_space, level, size, expected_names, variant
):
    """Check the samplers for MLMC-CV variant."""
    algo = MLMCMLCV(levels, uncertain_space, 1000.0, variant=variant)
    sampler = algo._samplers[level]
    input_samples, output_samples = sampler(2)
    names = [function.name for function in sampler._MonteCarloSampler__functions]
    assert names == expected_names
    assert input_samples.shape == (2, 1)
    assert output_samples.shape == (2, size)


@pytest.mark.parametrize(
    ("level", "variant", "expected"),
    [
        (0, MLMCMLCV.Variant.MLMC_MLCV, slice(0, 3)),
        (0, MLMCMLCV.Variant.MLMC_MLCV_0, slice(0, 2)),
        (0, MLMCMLCV.Variant.MLMC_CV, slice(0, 1)),
        (0, MLMCMLCV.Variant.MLMC_CV_0, slice(0, 1)),
        (1, MLMCMLCV.Variant.MLMC_MLCV, slice(0, 2)),
        (1, MLMCMLCV.Variant.MLMC_MLCV_0, slice(0, 1)),
        (1, MLMCMLCV.Variant.MLMC_CV, slice(0, 1)),
        (1, MLMCMLCV.Variant.MLMC_CV_0, slice(0, 0)),
        (2, MLMCMLCV.Variant.MLMC_MLCV, slice(0, 2)),
        (2, MLMCMLCV.Variant.MLMC_MLCV_0, slice(0, 1)),
        (2, MLMCMLCV.Variant.MLMC_CV, slice(1, 2)),
        (2, MLMCMLCV.Variant.MLMC_CV_0, slice(0, 0)),
    ],
)
def test_get_surrogate_positions(level, variant, expected):
    """Check get_surrogate_positions."""
    assert MLMCMLCV.get_surrogate_positions(level, 3, variant) == expected
