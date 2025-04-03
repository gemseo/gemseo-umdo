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
"""The GEMSEO-free version of the output data for the beam use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemseo.typing import RealArray


@dataclass
class BeamModelOutputData:
    """Output data of the beam model."""

    Ux: RealArray
    r"""The strain energy along the $x$-axis."""

    Uy: RealArray
    r"""The strain energy along the $y$-axis."""

    Uz: RealArray
    r"""The strain energy along the $z$-axis."""

    sigma: RealArray
    """The normal stress at the root section points."""

    tau: RealArray
    """The torsional stress at the root section points."""

    displ: RealArray
    """The displacements at the tip section points."""

    sigma_vm: RealArray
    """The von Mises stress at the root section points."""

    w: float
    """The weight of the beam."""

    yz_grid: RealArray
    r"""The $yz$-grid coordinates."""
