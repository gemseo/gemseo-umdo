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
"""Some variables of the GEMSEO-free version of the beam use case."""

from __future__ import annotations

from typing import Final
from typing import NamedTuple


class Variable(NamedTuple):
    """A variable of the beam use case."""

    name: str
    """The name of the variable."""

    value: float | None = None
    """The default value of the variable."""


b: Final[Variable] = Variable("b", 500.0)
"""The width of the beam."""

h: Final[Variable] = Variable("h", 800.0)
"""The height of the beam."""

t: Final[Variable] = Variable("t", 2.5)
"""The thickness of the beam."""

L: Final[Variable] = Variable("L", 5000.0)
"""The length of the beam."""

E: Final[Variable] = Variable("E", 73500.0)
"""The Young's modulus of the material."""

F: Final[Variable] = Variable("F", -200000.0)
"""The load applied to a point at the tip of the beam."""

alpha: Final[Variable] = Variable("alpha", 0.0)
r"""The angle between $-\vec{e}_z$ and $\vec{F}$ in $xy$-plane."""

beta: Final[Variable] = Variable("beta", 0.0)
r"""The angle between $-\vec{e}_z$ and $\vec{F}$ in $yz$-plane."""

dy: Final[Variable] = Variable("dy", 0.0)
r"""The $y$-coordinate of the point where the force is applied."""

dz: Final[Variable] = Variable("dz", 0.0)
r"""The $z$-coordinate of the point where the force is applied."""

rho: Final[Variable] = Variable("rho", 2.8e-6)
"""The density of the material."""

nu: Final[Variable] = Variable("nu", 0.33)
"""The Poisson's ratio."""

sigma_all: Final[Variable] = Variable("sigma_all", 300.0)
"""A constant used by the stress constraints."""

Rd: Final[Variable] = Variable("Rd", 180.0)
"""The design strength of the material.

The beam model does not use it.
"""

Ry: Final[Variable] = Variable("Ry", 600.0)
"""The yield strength of the material.

The beam model does not use it.
"""
