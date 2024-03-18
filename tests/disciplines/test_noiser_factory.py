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
"""Tests for NoiserFactory."""

from __future__ import annotations

import pytest

from gemseo_umdo.disciplines.multiplicative_noiser import MultiplicativeNoiser
from gemseo_umdo.disciplines.noiser_factory import NoiserFactory


def test_classes():
    """Check the classes that the factory can build."""
    class_names = NoiserFactory().class_names
    assert "AnalyticDiscipline" not in class_names
    assert {"MultiplicativeNoiser", "AdditiveNoiser"}.issubset(class_names)


@pytest.mark.parametrize("noiser_name", ["*", "MultiplicativeNoiser"])
def test_creation(noiser_name):
    """Check the creation of a BaseNoiser."""
    noiser = NoiserFactory().create(noiser_name, "foo", "noised_foo", "noise")
    assert isinstance(noiser, MultiplicativeNoiser)
    assert set(noiser.input_grammar.names) == {"foo", "noise"}
    assert set(noiser.output_grammar.names) == {"noised_foo"}
