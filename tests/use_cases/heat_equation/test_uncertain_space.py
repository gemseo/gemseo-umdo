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
from __future__ import annotations

import pytest
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTUniformDistribution_Settings
from numpy import pi

from gemseo_umdo.use_cases.heat_equation.uncertain_space import (
    HeatEquationUncertainSpace,
)


@pytest.mark.parametrize("nu_bounds", [None, (0.002, 0.004)])
def test_uncertain_space(nu_bounds):
    """Check the content of the uncertain space."""
    uncertain_space = RandomSpace()
    uncertain_space.add_variable(
        "X_1", OTUniformDistribution_Settings(minimum=-pi, maximum=pi)
    )
    uncertain_space.add_variable(
        "X_2", OTUniformDistribution_Settings(minimum=-pi, maximum=pi)
    )
    uncertain_space.add_variable(
        "X_3", OTUniformDistribution_Settings(minimum=-pi, maximum=pi)
    )
    if nu_bounds:
        minimum, maximum = nu_bounds
    else:
        minimum, maximum = 0.001, 0.009
    uncertain_space.add_variable(
        "X_4", OTUniformDistribution_Settings(minimum=minimum, maximum=maximum)
    )
    uncertain_space.add_variable(
        "X_5", OTUniformDistribution_Settings(minimum=-1.0, maximum=1.0)
    )
    uncertain_space.add_variable(
        "X_6", OTUniformDistribution_Settings(minimum=-1.0, maximum=1.0)
    )
    uncertain_space.add_variable(
        "X_7", OTUniformDistribution_Settings(minimum=-1.0, maximum=1.0)
    )

    if nu_bounds:
        he_uncertain_space = HeatEquationUncertainSpace(nu_bounds)
    else:
        he_uncertain_space = HeatEquationUncertainSpace()

    assert list(he_uncertain_space.variables) == list(uncertain_space.variables)
    for name in list(he_uncertain_space.variables):
        assert repr(he_uncertain_space.variables[name].distribution) == repr(
            uncertain_space.variables[name].distribution
        )
