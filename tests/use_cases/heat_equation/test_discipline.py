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
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.use_cases.heat_equation.discipline import HeatEquation


@pytest.fixture(scope="module")
def heat_equation() -> HeatEquation:
    """The heat equation discipline."""
    return HeatEquation()


def test_input_names(heat_equation):
    """Check the input names."""
    assert set(heat_equation.get_input_data_names()) == {f"X_{i}" for i in range(1, 8)}


def test_output_names(heat_equation):
    """Check the output names."""
    assert set(heat_equation.get_output_data_names()) == {"u", "u_mesh"}


def test_default_inputs(heat_equation):
    """Check the default inputs."""
    assert_equal(
        dict(heat_equation.default_inputs),
        {
            "X_1": array([0.0]),
            "X_2": array([0.0]),
            "X_3": array([0.0]),
            "X_4": array([0.005]),
            "X_5": array([0.0]),
            "X_6": array([0.0]),
            "X_7": array([0.0]),
        },
    )


def test_output_data(heat_equation):
    """Check the output values and sizes."""
    heat_equation.execute()
    u, u_mesh = heat_equation.get_local_data_by_name(["u", "u_mesh"])
    assert u.shape == (1,)
    assert u_mesh.shape == (heat_equation.configuration.mesh_size,)
    assert_almost_equal(u, -31.052594621006744)
    assert_almost_equal(u_mesh.mean(), -30.742068674796673)
    assert_almost_equal(u_mesh.std(), 15.259074356380882)


def test_expectation(heat_equation):
    """Check the analytical expression of the expectation."""
    assert heat_equation.configuration.expectation == pytest.approx(41.98447216482206)
