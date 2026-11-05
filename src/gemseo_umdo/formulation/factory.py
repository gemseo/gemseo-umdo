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
"""Factory of U-MDO formulations."""

from __future__ import annotations

from typing import ClassVar
from typing import Final

from gemseo.formulation.factory import MDOFormulationFactory

from gemseo_umdo.formulation.base_umdo_formulation import BaseUMDOFormulation


# TODO: API: rename to UMDOFormulationFactory
class UMDOFormulationsFactory(MDOFormulationFactory):
    """The factory of U-MDO formulations."""

    _class: ClassVar[type[BaseUMDOFormulation]] = BaseUMDOFormulation

    _package_names: ClassVar[tuple[str, ...]] = ("gemseo_umdo.formulation",)


umdo_formulation_factory: Final[UMDOFormulationsFactory] = UMDOFormulationsFactory()
"""The factory for ``BaseUMDOFormulation`` objects."""
