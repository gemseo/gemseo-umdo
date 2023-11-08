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
"""Test for the AdvancedBeamUncertainSpace."""

from __future__ import annotations

import pytest

from gemseo_umdo.use_cases.beam_model.advanced_uncertain_space import (
    AdvancedBeamUncertainSpace,
)
from gemseo_umdo.use_cases.beam_model.core.variables import b


@pytest.fixture(scope="module")
def uncertain_space() -> AdvancedBeamUncertainSpace:
    """The AdvancedBeamUncertainSpace."""
    return AdvancedBeamUncertainSpace()


def test_dimension(uncertain_space):
    """Check the dimension of the uncertain space."""
    assert uncertain_space.dimension == 11


def test_nominal_values():
    """Check the use of nominal values."""
    space = AdvancedBeamUncertainSpace(nominal_values={b.name: 510.0})
    assert space._AdvancedBeamUncertainSpace__nominal_values[b.name] == 510.0


@pytest.mark.parametrize(
    ("name", "distribution_class_name", "parameters"),
    [
        (
            "b",
            "OTUniformDistribution",
            {"maximum": 501.0, "minimum": 499.0},
        ),
        (
            "h",
            "OTUniformDistribution",
            {"maximum": 801.0, "minimum": 799.0},
        ),
        (
            "t",
            "OTUniformDistribution",
            {"maximum": 3.5, "minimum": 1.5},
        ),
        (
            "L",
            "OTUniformDistribution",
            {"maximum": 5001.0, "minimum": 4999.0},
        ),
        (
            "alpha",
            "OTNormalDistribution",
            {
                "lower_bound": -1.0,
                "mu": 0.0,
                "sigma": 0.3333333333333333,
                "upper_bound": 1.0,
            },
        ),
        (
            "beta",
            "OTNormalDistribution",
            {
                "lower_bound": -1.0,
                "mu": 0.0,
                "sigma": 0.3333333333333333,
                "upper_bound": 1.0,
            },
        ),
        (
            "dy",
            "OTNormalDistribution",
            {
                "lower_bound": -1.0,
                "mu": 0.0,
                "sigma": 0.3333333333333333,
                "upper_bound": 1.0,
            },
        ),
        (
            "dz",
            "OTNormalDistribution",
            {
                "lower_bound": -1.0,
                "mu": 0.0,
                "sigma": 0.3333333333333333,
                "upper_bound": 1.0,
            },
        ),
        (
            "E",
            "OTNormalDistribution",
            {
                "lower_bound": 73499.0,
                "mu": 73500.0,
                "sigma": 0.3333333333333333,
                "upper_bound": 73501.0,
            },
        ),
        (
            "Rd",
            "OTNormalDistribution",
            {
                "lower_bound": 179.0,
                "mu": 180.0,
                "sigma": 0.3333333333333333,
                "upper_bound": 181.0,
            },
        ),
        (
            "Ry",
            "OTNormalDistribution",
            {
                "lower_bound": 599.0,
                "mu": 600.0,
                "sigma": 0.3333333333333333,
                "upper_bound": 601.0,
            },
        ),
    ],
)
def test_variables(uncertain_space, name, distribution_class_name, parameters):
    """Check the probability distributions of the random variables."""
    distribution = uncertain_space[name]
    assert distribution.size == 1
    assert distribution.distribution == distribution_class_name
    assert distribution.parameters == pytest.approx(parameters)
