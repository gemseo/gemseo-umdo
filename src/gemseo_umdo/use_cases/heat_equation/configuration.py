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
r"""A configuration for the heat equation model.

[HeatEquationConfiguration][gemseo_umdo.use_cases.heat_equation.configuration.HeatEquationConfiguration]
is used by
[HeatEquationModel][gemseo_umdo.use_cases.heat_equation.model.HeatEquationModel];
read its docstring for more details.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from numpy import array
from numpy import exp
from numpy import linspace
from numpy import pi

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class HeatEquationConfiguration:
    """A configuration for the heat equation model."""

    mesh_size: int = 100
    """The number of equispaced spatial nodes."""

    n_modes: int = 21
    """The number of modes of the truncated Fourier expansion."""

    final_time: float = 0.5
    """The time of interest, denoted $t_f$."""

    nu_bounds: tuple[float, float] = (0.001, 0.009)
    """The bounds of the thermal diffusivity."""

    rod_length: float = 1.0
    """The length of the rod."""

    expectation: float = field(init=False)
    """The theoretical expectation of the integral of the temperature along the rod."""

    cost: int = field(init=False)
    """The evaluation cost of the
    [HeatEquationModel][gemseo_umdo.use_cases.heat_equation.model.HeatEquationModel]."""

    mesh: NDArray[float] = field(init=False)
    """The mesh for the
    [HeatEquationModel][gemseo_umdo.use_cases.heat_equation.model.HeatEquationModel]."""

    def __post_init__(self) -> None:
        self.expectation = self.__compute_expectation()
        self.cost = self.mesh_size * self.n_modes
        self.mesh = linspace(0.0, self.rod_length, num=self.mesh_size)

    def __compute_expectation(self) -> float:
        """Compute the expectation of the integral of the temperature along the rod.

        From Geraci et al., 2015 (Equation 5.3).

        Returns:
            The expectation of the integral of the temperature along the rod.
        """
        c = array([100 / pi, 98 / 12 / pi, 50 * 98 / 36 / pi, 50 * 98 / 84 / pi])
        pi2 = pi**2
        n2 = array([1, 3, 9, 21]) ** 2
        nu_delta = self.nu_bounds[1] - self.nu_bounds[0]
        h = (
            exp(-n2 * pi2 * self.nu_bounds[1] * self.final_time)
            * (1 - exp(n2 * pi2 * nu_delta * self.final_time))
            / (self.final_time * n2 * pi2 * nu_delta)
        )
        return -(c * h).sum()
