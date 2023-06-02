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
"""Tests for the BeamModel."""
from __future__ import annotations

from dataclasses import asdict

import pytest
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo_umdo.use_cases.beam_model.core.model import BeamModel
from gemseo_umdo.use_cases.beam_model.core.output_data import (
    BeamModelOutputData,
)
from numpy import array


@pytest.fixture(scope="module")
def model_output() -> BeamModelOutputData:
    """The default output data of the default beam model."""
    return BeamModel()()


@pytest.fixture(scope="module")
def custom_model_output() -> BeamModelOutputData:
    """A default output data of a custom beam model."""
    return BeamModel(n_y=2, n_z=4)()


@pytest.mark.parametrize(
    "name", ["displ", "sigma", "sigma_vm", "tau", "Ux", "Uy", "Uz"]
)
def test_output_shape_with_default_grid_size(model_output, name):
    """Check the output shapes with default grid size."""
    assert getattr(model_output, name).shape == (3, 3)


def test_grid_shape_with_default_grid_size(model_output):
    """Check the grid shape with default grid size."""
    assert model_output.yz_grid.shape == (9, 2)


@pytest.mark.parametrize(
    "name", ["displ", "sigma", "sigma_vm", "tau", "Ux", "Uy", "Uz"]
)
def test_output_shape_with_custom_grid_size(custom_model_output, name):
    """Check the output shapes with custom grid size."""
    assert getattr(custom_model_output, name).shape == (4, 2)


def test_grid_shape_with_custom_grid_size(custom_model_output):
    """Check the grid shape with default grid size."""
    assert custom_model_output.yz_grid.shape == (8, 2)


def test_default_output_data(model_output):
    """Check the values of the default output data."""
    expected = {
        "Ux": array(
            [
                [-22.41927948, -22.41927948, -22.41927948],
                [0.0, 0.0, 0.0],
                [22.41927948, 22.41927948, 22.41927948],
            ]
        ),
        "Uy": array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "Uz": array(
            [
                [-186.82732897, -186.82732897, -186.82732897],
                [-186.82732897, -186.82732897, -186.82732897],
                [-186.82732897, -186.82732897, -186.82732897],
            ]
        ),
        "sigma": array(
            [
                [-659.1268166, -659.1268166, -659.1268166],
                [0.0, 0.0, 0.0],
                [659.1268166, 659.1268166, 659.1268166],
            ]
        ),
        "tau": array(
            [
                [32.68882803, 0.0, -32.68882803],
                [59.0539007, 0.0, -59.0539007],
                [32.68882803, 0.0, -32.68882803],
            ]
        ),
        "displ": array(
            [
                [188.16767773, 188.16767773, 188.16767773],
                [186.82732897, 186.82732897, 186.82732897],
                [188.16767773, 188.16767773, 188.16767773],
            ]
        ),
        "sigma_vm": array(
            [
                [661.55410874, 659.1268166, 661.55410874],
                [102.28435639, 0.0, 102.28435639],
                [661.55410874, 659.1268166, 661.55410874],
            ]
        ),
        "w": 90.64999999999999,
        "yz_grid": array(
            [
                [-250.0, -400.0],
                [0.0, -400.0],
                [250.0, -400.0],
                [-250.0, 0.0],
                [0.0, 0.0],
                [250.0, 0.0],
                [-250.0, 400.0],
                [0.0, 400.0],
                [250.0, 400.0],
            ]
        ),
    }
    assert compare_dict_of_arrays(asdict(model_output), expected, tolerance=0.01)
