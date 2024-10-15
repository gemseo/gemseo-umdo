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
"""Tests for SpringMassDiscipline."""

from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo_umdo.use_cases.spring_mass_model.discipline import SpringMassDiscipline


@pytest.fixture(scope="module")
def discipline() -> SpringMassDiscipline:
    """The spring-mass discipline."""
    return SpringMassDiscipline()


def test_default_inputs(discipline):
    """Check the default inputs of the spring-mass discipline."""
    assert_equal(dict(discipline.default_input_data), {"stiffness": array([2.25])})


def test_name(discipline):
    """Check the name of the spring-mass discipline."""
    assert discipline.name == "SpringMassDiscipline(0.1)"


def test_input_names(discipline):
    """Check the input names of the spring-mass discipline."""
    assert set(discipline.io.input_grammar.names) == {"stiffness"}


def test_output_names(discipline):
    """Check the output names of the spring-mass discipline."""
    assert set(discipline.io.output_grammar.names) == {
        "displacement",
        "max_displacement",
    }


def test_cost(discipline):
    """Check the cost of the spring-mass discipline."""
    assert discipline.cost == 10.0


def test_output_data_with_default_settings(discipline):
    """Check the data outputted by the spring-mass discipline with default settings."""
    discipline.execute()
    d = discipline.io.data["displacement"]
    max_d = discipline.io.data["max_displacement"]
    assert max_d.size == 1
    assert d.size == 100
    assert_almost_equal(max_d, 13.07, decimal=2)
    assert_almost_equal(d.mean(), 6.70, decimal=2)
    assert_almost_equal(d.std(), 4.56, decimal=2)


@pytest.mark.parametrize(
    ("name", "value", "size", "max_d", "mean", "std", "cost"),
    [
        ("mass", 2.0, 100, 17.42, 9.41, 6.25, 10.0),
        ("initial_state", (0.2, 0.3), 100, 12.87, 6.70, 4.42, 10.0),
        ("initial_time", 5.0, 50, 13.06, 6.70, 4.56, 10.0),
        ("final_time", 20.0, 200, 13.07, 6.69, 4.58, 10.0),
        ("time_step", 0.05, 200, 13.07, 6.70, 4.56, 20.0),
        ("gravity", 9.7, 100, 12.93, 6.63, 4.51, 10.0),
    ],
)
def test_output_data_with_custom_settings(name, value, size, max_d, mean, std, cost):
    """Check the data outputted by the spring-mass discipline with custom settings."""
    discipline = SpringMassDiscipline(**{name: value})
    assert discipline.cost == cost
    discipline.execute()
    d = discipline.io.data["displacement"]
    d_max = discipline.io.data["max_displacement"]
    assert d_max.size == 1
    assert d.size == size
    assert_almost_equal(max_d, array([d_max]), decimal=2)
    assert_almost_equal(d.mean(), mean, decimal=2)
    assert_almost_equal(d.std(), std, decimal=2)


def test_output_data_with_custom_stiffness(discipline):
    """Check the data outputted by the spring-mass discipline with custom stiffness."""
    discipline.execute({"stiffness": array([2.5])})
    d = discipline.io.data["displacement"]
    d_max = discipline.io.data["max_displacement"]
    assert_almost_equal(d_max, array([11.76]), decimal=2)
    assert_almost_equal(d.mean(), 5.73, decimal=2)
    assert_almost_equal(d.std(), 4.21, decimal=2)
