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
from gemseo.util.testing.helper import assert_exception

from gemseo_umdo.problem.beam_model.advanced_uncertain_space import (
    AdvancedBeamUncertainSpace,
)


@pytest.fixture(scope="module")
def uncertain_space() -> AdvancedBeamUncertainSpace:
    """The AdvancedBeamUncertainSpace."""
    return AdvancedBeamUncertainSpace()


def test_dimension(uncertain_space):
    """Check the dimension of the uncertain space."""
    assert uncertain_space.dimension == 11


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nominal_values": {"foo": 1.0}},
        {"bar": 2.0},
        {"nominal_values": {"foo": 1.0}, "bar": 2.0},
    ],
    ids=["nominal-value", "dispersion", "several"],
)
def test_unknown_name(kwargs, snapshot):
    """Check that an unknown variable name raises an error."""
    with assert_exception(ValueError, snapshot):
        AdvancedBeamUncertainSpace(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t": 2.5},
        {"nominal_values": {"t": 0.5}, "t": 1.0},
        {"nominal_values": {"E": 0.0}},
        {"nominal_values": {"E": -1.0}, "E": 0.5},
        {"nominal_values": {"Rd": 1.0}, "Rd": 3.0},
    ],
    ids=[
        "zero-lower-bound",
        "negative-lower-bound",
        "zero-nominal-value",
        "negative-nominal-value",
        "normal-negative-lower-bound",
    ],
)
def test_invalid_support(kwargs, snapshot):
    """Check that a positive variable with a non-positive support raises an error."""
    with assert_exception(ValueError, snapshot):
        AdvancedBeamUncertainSpace(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"b": 0.0}, {"alpha": -1.0}],
    ids=["zero", "negative"],
)
def test_invalid_dispersion(kwargs, snapshot):
    """Check that a non-positive dispersion raises an error."""
    with assert_exception(ValueError, snapshot):
        AdvancedBeamUncertainSpace(**kwargs)


@pytest.mark.parametrize(
    ("name", "repr_"),
    [
        ("b", "Uniform(lower=475.0, upper=525.0)"),
        ("h", "Uniform(lower=760.0, upper=840.0)"),
        ("t", "Uniform(lower=2.375, upper=2.625)"),
        ("L", "Uniform(lower=4750.0, upper=5250.0)"),
        ("alpha", "Normal(mu=0.0, sigma=0.005817764173314432)"),
        ("beta", "Normal(mu=0.0, sigma=0.005817764173314432)"),
        ("dy", "Normal(mu=0.0, sigma=8.333333333333334)"),
        ("dz", "Normal(mu=0.0, sigma=13.333333333333334)"),
        ("E", "Normal(mu=73500.0, sigma=1225.0)"),
        ("Rd", "Normal(mu=180.0, sigma=3.0)"),
        ("Ry", "Normal(mu=600.0, sigma=10.0)"),
    ],
)
def test_variables(uncertain_space, name, repr_):
    """Check the default probability distributions of the random variables."""
    assert repr(uncertain_space.variables[name].distribution) == repr_


@pytest.mark.parametrize(
    ("kwargs", "name", "repr_", "bounds"),
    [
        (
            {"nominal_values": {"b": 510.0}},
            "b",
            "Uniform(lower=484.5, upper=535.5)",
            (484.5, 535.5),
        ),
        (
            {"nominal_values": {"E": 70000.0}},
            "E",
            "Normal(mu=70000.0, sigma=1166.6666666666667)",
            (66500.0, 73500.0),
        ),
        (
            {"nominal_values": {"alpha": 0.1}},
            "alpha",
            "Normal(mu=0.1, sigma=0.005817764173314432)",
            (0.08254670599, 0.11745329401),
        ),
        (
            {"nominal_values": {"alpha": -0.1}},
            "alpha",
            "Normal(mu=-0.1, sigma=0.005817764173314432)",
            (-0.11745329401, -0.08254670599),
        ),
        (
            {"nominal_values": {"dy": 1.0}},
            "dy",
            "Normal(mu=1.0, sigma=8.333333333333334)",
            (-24.0, 26.0),
        ),
        (
            {"nominal_values": {"b": 600.0}},
            "dy",
            "Normal(mu=0.0, sigma=10.0)",
            (-30.0, 30.0),
        ),
        ({"b": 2.0}, "b", "Uniform(lower=498.0, upper=502.0)", (498.0, 502.0)),
        ({"alpha": 3.0}, "alpha", "Normal(mu=0.0, sigma=1.0)", (-3.0, 3.0)),
    ],
    ids=[
        "uniform-nominal",
        "normal-nominal",
        "positive-angle-nominal",
        "negative-angle-nominal",
        "non-zero-coordinate-nominal",
        "coordinate-from-b",
        "uniform-dispersion",
        "normal-dispersion",
    ],
)
def test_custom_variables(kwargs, name, repr_, bounds):
    """Check the probability distributions from user nominal values and dispersions.

    A missing dispersion is relative to the nominal value for a positive variable,
    1 degree for an angle
    and relative to the nominal value of `b` for `dy`, whatever its nominal value.
    A truncated normal is truncated at three standard deviations.
    """
    distribution = AdvancedBeamUncertainSpace(**kwargs).variables[name].distribution
    assert repr(distribution) == repr_
    assert distribution.math_lower_bound == pytest.approx(bounds[0])
    assert distribution.math_upper_bound == pytest.approx(bounds[1])
