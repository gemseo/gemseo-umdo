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
"""Tests for HeatEquationModel."""
from __future__ import annotations

import pytest
from numpy import array
from numpy import isnan
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.use_cases.heat_equation.configuration import HeatEquationConfiguration
from gemseo_umdo.use_cases.heat_equation.model import HeatEquationModel


@pytest.fixture(scope="module")
def heat_equation_model() -> HeatEquationModel:
    """The heat equation model."""
    return HeatEquationModel()


def test_default_configuration(heat_equation_model):
    """Check the HeatEquationConfiguration attached to the default HeatEquationModel."""
    expected_configuration = HeatEquationConfiguration()
    configuration = heat_equation_model.configuration
    assert configuration.n_modes == expected_configuration.n_modes
    assert configuration.final_time == expected_configuration.final_time
    assert configuration.nu_bounds == expected_configuration.nu_bounds
    assert configuration.rod_length == expected_configuration.rod_length
    assert configuration.expectation == expected_configuration.expectation
    assert configuration.cost == expected_configuration.cost
    assert_equal(configuration.mesh_size, expected_configuration.mesh_size)


def test_custom_configuration():
    """Check the HeatEquationConfiguration attached to a custom HeatEquationModel."""
    configuration = HeatEquationModel(
        mesh_size=1, n_modes=2, final_time=3.0, nu_bounds=(4.0, 5.0), rod_length=6.0
    ).configuration
    expected_configuration = HeatEquationConfiguration(
        mesh_size=1, n_modes=2, final_time=3.0, nu_bounds=(4.0, 5.0), rod_length=6.0
    )
    assert configuration.n_modes == expected_configuration.n_modes
    assert configuration.final_time == expected_configuration.final_time
    assert configuration.nu_bounds == expected_configuration.nu_bounds
    assert configuration.rod_length == expected_configuration.rod_length
    assert isnan(configuration.expectation)
    assert isnan(expected_configuration.expectation)
    assert configuration.cost == expected_configuration.cost
    assert_equal(configuration.mesh_size, expected_configuration.mesh_size)


def test_default_output_data(heat_equation_model):
    """Check the output values and sizes with default model."""
    u, u_mesh = heat_equation_model()
    assert u.shape == ()
    assert u_mesh.shape == (heat_equation_model.configuration.mesh_size,)
    assert_almost_equal(u, -31.052594621006744)
    assert_almost_equal(u_mesh.mean(), -30.742068674796673)
    assert_almost_equal(u_mesh.std(), 15.259074356380882)


@pytest.mark.parametrize("n_samples", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_custom_output_data(heat_equation_model, n_samples, batch_size):
    """Check the output values and sizes with custom model."""
    samples = array([[0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0]] * n_samples)
    u, u_mesh = heat_equation_model(samples, batch_size=batch_size)
    assert u.shape == (n_samples,)
    assert u_mesh.shape == (n_samples, heat_equation_model.configuration.mesh_size)
    assert_almost_equal(u, -31.052594621006744)
    assert_almost_equal(u_mesh.mean(), -30.742068674796673)
    assert_almost_equal(u_mesh.std(), 15.259074356380882)


def test_compute_taylor(heat_equation_model):
    """Check compute_taylor."""
    output_data = heat_equation_model.compute_taylor(
        array([[0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0]] * 3)
    )
    assert_almost_equal(output_data, array([[-31.0525946]] * 3))
