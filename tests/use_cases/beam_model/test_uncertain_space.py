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


@pytest.mark.parametrize("uncertain_space", (False, True), indirect=True)
def test_dimension(uncertain_space):
    """Check the dimension of the uncertain space."""
    assert uncertain_space[0].dimension == 3


@pytest.mark.parametrize("uncertain_space", (False, True), indirect=True)
@pytest.mark.parametrize(
    "name,parameters",
    [
        (
            "F",
            (
                {"mu": -200000.0, "sigma": 6666.67},
                {"maximum": -180000.0, "minimum": -220000.00000000003},
            ),
        ),
        (
            "E",
            (
                {"mu": 73500.0, "sigma": 1225.0},
                {"maximum": 77175.0, "minimum": 69825.0},
            ),
        ),
        (
            "sigma_all",
            ({"mu": 300.0, "sigma": 5.0}, {"maximum": 315.0, "minimum": 285.0}),
        ),
    ],
)
def test_variables(uncertain_space, name, parameters):
    """Check the probability distributions of the random variables."""
    distribution = uncertain_space[0][name]
    assert distribution.size == 1
    if uncertain_space[1]:
        assert distribution.distribution == "OTUniformDistribution"
        assert distribution.parameters == pytest.approx(parameters[1])
    else:
        assert distribution.distribution == "OTNormalDistribution"
        assert distribution.parameters == pytest.approx(parameters[0])
