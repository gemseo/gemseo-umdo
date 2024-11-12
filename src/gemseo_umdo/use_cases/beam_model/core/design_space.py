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
"""The GEMSEO-free version of the design space for the beam use case."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from gemseo_umdo.use_cases.beam_model.core.variables import h
from gemseo_umdo.use_cases.beam_model.core.variables import t


class DesignVariable(NamedTuple):
    """A design variable."""

    name: str
    """The name of the design variable."""

    l_b: float
    """The lower bound of the design variable."""

    u_b: float
    """The upper bound of the design variable."""

    value: float
    """The current value of the design variable."""


class BeamDesignVariables(Enum):
    """The design variables for the beam use case."""

    h = DesignVariable(h.name, 500, 800, h.value)
    """The height of the beam."""

    t = DesignVariable(t.name, 2, 10, t.value)
    """The thickness of the beam."""
