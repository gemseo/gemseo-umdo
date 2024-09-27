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
    ("name", "repr_"),
    [
        ("b", "Uniform(lower=499.0, upper=501.0)"),
        ("h", "Uniform(lower=799.0, upper=801.0)"),
        ("t", "Uniform(lower=1.5, upper=3.5)"),
        ("L", "Uniform(lower=4999.0, upper=5001.0)"),
        ("alpha", "Normal(mu=0.0, sigma=0.3333333333333333)"),
        ("beta", "Normal(mu=0.0, sigma=0.3333333333333333)"),
        ("dy", "Normal(mu=0.0, sigma=0.3333333333333333)"),
        ("dz", "Normal(mu=0.0, sigma=0.3333333333333333)"),
        ("E", "Normal(mu=73500.0, sigma=0.3333333333333333)"),
        ("Rd", "Normal(mu=180.0, sigma=0.3333333333333333)"),
        ("Ry", "Normal(mu=600.0, sigma=0.3333333333333333)"),
    ],
)
def test_variables(uncertain_space, name, repr_):
    """Check the probability distributions of the random variables."""
    assert repr(uncertain_space.distributions[name]) == repr_
