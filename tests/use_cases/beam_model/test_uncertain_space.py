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
"""Test for the BeamUncertainSpace."""

from __future__ import annotations

import pytest

from gemseo_umdo.use_cases.beam_model.uncertain_space import BeamUncertainSpace


@pytest.fixture(scope="module")
def uncertain_space(request) -> tuple[BeamUncertainSpace, bool]:
    """The BeamUncertainSpace."""
    return BeamUncertainSpace(uniform=request.param), request.param


@pytest.mark.parametrize("uncertain_space", [False, True], indirect=True)
def test_dimension(uncertain_space):
    """Check the dimension of the uncertain space."""
    assert uncertain_space[0].dimension == 3


@pytest.mark.parametrize("uncertain_space", [False, True], indirect=True)
@pytest.mark.parametrize(
    ("name", "repr_"),
    [
        (
            "F",
            (
                "Normal(mu=-200000.0, sigma=6666.666666666667)",
                "Uniform(lower=-220000.00000000003, upper=-180000.0)",
            ),
        ),
        (
            "E",
            (
                "Normal(mu=73500.0, sigma=1225.0)",
                "Uniform(lower=69825.0, upper=77175.0)",
            ),
        ),
        (
            "sigma_all",
            ("Normal(mu=300.0, sigma=5.0)", "Uniform(lower=285.0, upper=315.0)"),
        ),
    ],
)
def test_variables(uncertain_space, name, repr_):
    """Check the probability distributions of the random variables."""
    joint_distribution = uncertain_space[0].distributions[name]
    assert joint_distribution.dimension == 1
    assert repr(joint_distribution) == repr_[int(bool(uncertain_space[1]))]
