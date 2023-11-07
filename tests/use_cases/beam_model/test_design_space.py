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
"""Test for the BeamDesignSpace."""
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from numpy import array

from gemseo_umdo.use_cases.beam_model.core.design_space import BeamDesignVariables
from gemseo_umdo.use_cases.beam_model.design_space import BeamDesignSpace


@pytest.fixture(scope="module")
def design_space() -> BeamDesignSpace:
    """The BeamDesignSpace."""
    return BeamDesignSpace()


def test_dimension(design_space):
    """Check the dimension of the design space."""
    assert design_space.dimension == 2


@pytest.mark.parametrize("variable", [BeamDesignVariables.h, BeamDesignVariables.t])
def test_variables(design_space, variable):
    """Check the properties of the design variables."""
    assert design_space[variable.value.name] == DesignSpace.DesignVariable(
        1,
        "float",
        array([variable.value.l_b]),
        array([variable.value.u_b]),
        array([variable.value.value]),
    )
