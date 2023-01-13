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
"""Formulate a multidisciplinary design problem under uncertainty."""
from __future__ import annotations

from gemseo.core.factory import Factory
from gemseo.formulations.formulations_factory import MDOFormulationsFactory

from gemseo_umdo.formulations.formulation import UMDOFormulation


class UMDOFormulationsFactory(MDOFormulationsFactory):
    """The factory of :class:`.UMDOFormulation`."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.factory = Factory(UMDOFormulation, ("gemseo_umdo.formulations",))
