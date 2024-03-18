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
"""Tests for AdditiveNoiser."""

from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo_umdo.disciplines.additive_noiser import AdditiveNoiser
from gemseo_umdo.disciplines.noiser_factory import NoiserFactory


@pytest.fixture()
def additive_noiser() -> AdditiveNoiser:
    """An additive noiser."""
    return AdditiveNoiser("foo", "noised_foo", "noise")


def test_short_name():
    """Check the short name."""
    assert AdditiveNoiser.SHORT_NAME == "+"
    noiser = NoiserFactory().create(
        "+",
        variable_name="foo",
        noised_variable_name="noised_foo",
        uncertain_variable_name="noise",
    )
    assert isinstance(noiser, AdditiveNoiser)


@pytest.mark.parametrize(
    ("foo", "noise", "noised_foo"),
    [
        (array([2]), array([3]), array([5])),
        (array([2, 2]), array([3]), array([5, 5])),
        (array([2, 2]), array([3, 4]), array([5, 6])),
    ],
)
def test_execute(additive_noiser, foo, noise, noised_foo):
    """Check the execution."""
    additive_noiser.execute({"foo": foo, "noise": noise})
    assert_equal(additive_noiser.local_data["noised_foo"], noised_foo)


@pytest.mark.parametrize(
    ("foo", "noise", "jac"),
    [
        (
            array([2]),
            array([3]),
            {"noised_foo": {"foo": array([[1]]), "noise": array([[1]])}},
        ),
        (
            array([2, 2]),
            array([3]),
            {
                "noised_foo": {
                    "foo": array([[1, 0], [0, 1]]),
                    "noise": array([[1], [1]]),
                }
            },
        ),
        (
            array([2, 2]),
            array([3, 4]),
            {
                "noised_foo": {
                    "foo": array([[1, 0], [0, 1]]),
                    "noise": array([[1, 0], [0, 1]]),
                }
            },
        ),
    ],
)
def test_jacobian(additive_noiser, foo, noise, jac):
    """Check the computation of the Jacobian."""
    additive_noiser.linearize({"foo": foo, "noise": noise}, compute_all_jacobians=True)
    assert_equal(additive_noiser.jac["noised_foo"], jac["noised_foo"])
