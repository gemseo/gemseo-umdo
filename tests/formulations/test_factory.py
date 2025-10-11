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

from gemseo.formulations.factory import MDOFormulationFactory

from gemseo_umdo.formulations.factory import UMDO_FORMULATION_FACTORY


def test_inheritance():
    """Check that the factory is also a MDOFormulationsFactory."""
    assert isinstance(UMDO_FORMULATION_FACTORY, MDOFormulationFactory)


def test_mdo_formulation():
    """Check that UMDOFormulationsFactory does not contain MDO formulations."""
    mdo_formulations = MDOFormulationFactory().class_names
    assert not set(mdo_formulations).intersection(
        set(UMDO_FORMULATION_FACTORY.class_names)
    )


def test_u_formulation():
    """Check the method is_available of the factory."""
    assert UMDO_FORMULATION_FACTORY.is_available("Sampling")
