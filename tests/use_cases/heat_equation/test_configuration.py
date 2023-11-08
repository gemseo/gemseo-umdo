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
"""Tests for HeatEquationConfiguration."""

from __future__ import annotations

import pytest
from numpy import linspace
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo_umdo.use_cases.heat_equation.configuration import HeatEquationConfiguration


def test_default_configuration_and_cost():
    """Check the default configuration."""
    default_configuration = HeatEquationConfiguration()
    expected_configuration = HeatEquationConfiguration(
        mesh_size=100,
        n_modes=21,
        final_time=0.5,
        nu_bounds=(0.001, 0.009),
        rod_length=1.0,
    )
    assert default_configuration.n_modes == expected_configuration.n_modes
    assert default_configuration.final_time == expected_configuration.final_time
    assert default_configuration.nu_bounds == expected_configuration.nu_bounds
    assert default_configuration.rod_length == expected_configuration.rod_length
    assert default_configuration.expectation == expected_configuration.expectation
    assert default_configuration.cost == expected_configuration.cost
    assert_equal(default_configuration.mesh_size, expected_configuration.mesh_size)


def test_custom_configuration():
    """Check a custom configuration."""
    custom_configuration = HeatEquationConfiguration(
        mesh_size=100,
        n_modes=20,
        final_time=0.5,
        nu_bounds=(0.001, 0.009),
        rod_length=1.2,
    )
    assert (
        custom_configuration.cost
        == custom_configuration.mesh_size * custom_configuration.n_modes
    )
    assert custom_configuration.expectation == pytest.approx(41.98, abs=1e-2)
    assert_allclose(
        custom_configuration.mesh,
        linspace(
            0.0, custom_configuration.rod_length, num=custom_configuration.mesh_size
        ),
    )
