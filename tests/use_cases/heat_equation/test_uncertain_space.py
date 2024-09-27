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
from gemseo.algos.parameter_space import ParameterSpace
from numpy import pi

from gemseo_umdo.use_cases.heat_equation.uncertain_space import (
    HeatEquationUncertainSpace,
)


@pytest.mark.parametrize("nu_bounds", [None, (0.002, 0.004)])
def test_uncertain_space(nu_bounds):
    """Check the content of the uncertain space."""
    distribution_name = "OTUniformDistribution"
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable(
        "X_1", distribution_name, minimum=-pi, maximum=pi
    )
    uncertain_space.add_random_variable(
        "X_2", distribution_name, minimum=-pi, maximum=pi
    )
    uncertain_space.add_random_variable(
        "X_3", distribution_name, minimum=-pi, maximum=pi
    )
    if nu_bounds:
        minimum, maximum = nu_bounds
    else:
        minimum, maximum = 0.001, 0.009
    uncertain_space.add_random_variable(
        "X_4", distribution_name, minimum=minimum, maximum=maximum
    )
    uncertain_space.add_random_variable(
        "X_5", distribution_name, minimum=-1.0, maximum=1.0
    )
    uncertain_space.add_random_variable(
        "X_6", distribution_name, minimum=-1.0, maximum=1.0
    )
    uncertain_space.add_random_variable(
        "X_7", distribution_name, minimum=-1.0, maximum=1.0
    )

    if nu_bounds:
        he_uncertain_space = HeatEquationUncertainSpace(nu_bounds)
    else:
        he_uncertain_space = HeatEquationUncertainSpace()

    assert he_uncertain_space.variable_names == uncertain_space.variable_names
    for name in he_uncertain_space.variable_names:
        assert repr(he_uncertain_space.distributions[name]) == repr(
            uncertain_space.distributions[name]
        )
