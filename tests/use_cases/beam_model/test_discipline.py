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
"""Tests for the Beam discipline."""

from __future__ import annotations

from dataclasses import asdict

import pytest
from gemseo.utils.comparisons import compare_dict_of_arrays
from numpy import array
from numpy import atleast_1d

from gemseo_umdo.use_cases.beam_model.core.model import BeamModel
from gemseo_umdo.use_cases.beam_model.core.variables import E
from gemseo_umdo.use_cases.beam_model.core.variables import F
from gemseo_umdo.use_cases.beam_model.core.variables import L
from gemseo_umdo.use_cases.beam_model.core.variables import alpha
from gemseo_umdo.use_cases.beam_model.core.variables import b
from gemseo_umdo.use_cases.beam_model.core.variables import beta
from gemseo_umdo.use_cases.beam_model.core.variables import dy
from gemseo_umdo.use_cases.beam_model.core.variables import dz
from gemseo_umdo.use_cases.beam_model.core.variables import h
from gemseo_umdo.use_cases.beam_model.core.variables import nu
from gemseo_umdo.use_cases.beam_model.core.variables import rho
from gemseo_umdo.use_cases.beam_model.core.variables import t
from gemseo_umdo.use_cases.beam_model.discipline import Beam


@pytest.fixture(scope="module")
def discipline() -> Beam:
    """The Beam discipline."""
    disc = Beam()
    disc.execute()
    return disc


@pytest.fixture(scope="module")
def custom_discipline() -> Beam:
    """The Beam discipline with a custom grid size."""
    disc = Beam(n_y=4, n_z=2)
    disc.execute()
    return disc


def test_input_names(discipline):
    """Check the names of the inputs."""
    assert list(discipline.input_grammar.names) == [
        "b",
        "h",
        "t",
        "L",
        "E",
        "alpha",
        "beta",
        "dy",
        "dz",
        "rho",
        "F",
        "nu",
    ]


def test_output_names(discipline):
    """Check the names of the outputs."""
    assert list(discipline.output_grammar.names) == [
        "Ux",
        "Uy",
        "Uz",
        "sigma",
        "tau",
        "displ",
        "sigma_vm",
        "w",
        "yz_grid",
    ]


def test_default_inputs(discipline):
    """Check the default values of the inputs."""
    assert compare_dict_of_arrays(
        discipline.default_inputs,
        {
            variable.name: array([variable.value])
            for variable in [b, h, t, L, E, alpha, beta, dy, dz, rho, F, nu]
        },
    )


def test_default_outputs(discipline):
    """Check the default values of the outputs."""
    assert compare_dict_of_arrays(
        discipline.get_output_data(),
        {k: atleast_1d(v).ravel() for k, v in asdict(BeamModel()()).items()},
    )


def test_ny_nz(custom_discipline):
    """Check the use of a custom grid size."""
    for name, value in custom_discipline.local_data.items():
        assert value.shape == (16,) if name == "yz_grid" else (8,)
