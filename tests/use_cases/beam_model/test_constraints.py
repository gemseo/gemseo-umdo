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
"""Tests for the BeamConstraints discipline."""
from __future__ import annotations

import pytest
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo_umdo.use_cases.beam_model.constraints import BeamConstraints
from numpy import array
from numpy.testing import assert_almost_equal


@pytest.fixture(scope="module")
def discipline() -> BeamConstraints:
    """The BeamConstraints discipline."""
    disc = BeamConstraints()
    disc.execute()
    return disc


def test_input_names(discipline):
    """Check the names of the inputs."""
    assert list(discipline.input_grammar.names) == ["displ", "sigma_vm", "sigma_all"]


def test_output_names(discipline):
    """Check the names of the outputs."""
    assert list(discipline.output_grammar.names) == ["c_displ", "c_stress"]


def test_default_inputs(discipline):
    """Check the default values of the inputs."""
    expected = {
        "sigma_all": array([300.0]),
        "sigma_vm": array([300.0]),
        "displ": array([100.0]),
    }
    assert compare_dict_of_arrays(discipline.default_inputs, expected, tolerance=0.01)


def test_default_outputs(discipline):
    """Check the default values of the outputs."""
    assert_almost_equal(
        discipline.local_data["c_stress"],
        discipline.local_data["sigma_all"] / (discipline.local_data["sigma_vm"] + 1.0),
    )
    assert_almost_equal(
        discipline.local_data["c_displ"],
        100.0 / (discipline.local_data["displ"] + 0.1),
    )
