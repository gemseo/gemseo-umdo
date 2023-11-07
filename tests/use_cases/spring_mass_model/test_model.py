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
"""Tests for SpringMassModel."""
from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_almost_equal

from gemseo_umdo.use_cases.spring_mass_model.model import SpringMassModel


@pytest.fixture(scope="module")
def model() -> SpringMassModel:
    """The spring-mass model."""
    return SpringMassModel()


def test_cost(model):
    """Check the cost of the spring-mass model."""
    assert model.cost == 10.0


def test_output_data_with_default_settings(model):
    """Check the data outputted by the spring-mass model with default settings."""
    d, max_d = model()
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
    """Check the data outputted by the spring-mass model with custom settings."""
    model = SpringMassModel(**{name: value})
    assert model.cost == cost
    d, d_max = model()
    assert d_max.size == 1
    assert d.size == size
    assert_almost_equal(max_d, array([max_d]), decimal=2)
    assert_almost_equal(d.mean(), mean, decimal=2)
    assert_almost_equal(d.std(), std, decimal=2)


def test_output_data_with_custom_stiffness(model):
    """Check the data outputted by the spring-mass model with custom stiffness."""
    d, d_max = model(2.5)
    assert_almost_equal(d_max, 11.76, decimal=2)
    assert_almost_equal(d.mean(), 5.73, decimal=2)
    assert_almost_equal(d.std(), 4.21, decimal=2)
