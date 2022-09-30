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
from __future__ import annotations

import pytest
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo_umdo.formulations.factory import UMDOFormulationsFactory


@pytest.fixture
def factory() -> UMDOFormulationsFactory:
    """The factory of UMDOFormulation."""
    return UMDOFormulationsFactory()


def test_inheritance(factory):
    """Check that the factory is also a MDOFormulationsFactory."""
    assert isinstance(factory, MDOFormulationsFactory)


def test_mdo_formulation(factory):
    """Check that UMDOFormulationsFactory does not contain MDO formulations."""
    mdo_formulations = MDOFormulationsFactory().formulations
    assert not set(mdo_formulations).intersection(set(factory.formulations))


def test_u_formulation(factory):
    """Check the method is_available of the factory."""
    assert factory.is_available("Sampling")
