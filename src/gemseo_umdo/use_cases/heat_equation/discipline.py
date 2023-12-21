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
r"""The heat equation discipline.

This discipline wraps
the [HeatEquationModel][gemseo_umdo.use_cases.heat_equation.model.HeatEquationModel];
read its docstring for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.discipline import MDODiscipline
from numpy import array

from gemseo_umdo.use_cases.heat_equation.model import HeatEquationModel

if TYPE_CHECKING:
    from gemseo_umdo.use_cases.heat_equation.configuration import (
        HeatEquationConfiguration,
    )


class HeatEquation(MDODiscipline):
    """The discipline computing the temperature averaged over the rod at final time."""

    def __init__(
        self,
        mesh_size: int = 100,
        n_modes: int = 21,
        final_time: float = 0.5,
        nu_bounds: tuple[float, float] = (0.001, 0.009),
        rod_length: float = 1.0,
    ) -> None:
        """
        Args:
            mesh_size: The number of equispaced spatial nodes.
            n_modes: The number of modes of the truncated Fourier expansion.
            final_time: The time of interest.
            nu_bounds: The bounds of the thermal diffusivity.
            rod_length: The length of the rod.
        """  # noqa: D205 D212 D415
        super().__init__(name=f"{self.__class__.__name__}({mesh_size})")
        self.input_grammar.update_from_names([f"X_{i}" for i in range(1, 8)])
        self.output_grammar.update_from_names(["u", "u_mesh"])
        self.__heat_equation_model = HeatEquationModel(
            mesh_size=mesh_size,
            n_modes=n_modes,
            final_time=final_time,
            nu_bounds=nu_bounds,
            rod_length=rod_length,
        )
        self.default_inputs = {
            "X_1": array([0.0]),
            "X_2": array([0.0]),
            "X_3": array([0.0]),
            "X_4": array([0.005]),
            "X_5": array([0.0]),
            "X_6": array([0.0]),
            "X_7": array([0.0]),
        }

    @property
    def configuration(self) -> HeatEquationConfiguration:
        """The configuration."""
        return self.__heat_equation_model.configuration

    def _run(self) -> None:
        """Compute the temperature at final time with a truncated Fourier expansion.

        From Geraci et al., 2015 (Equation 5.4).
        """
        u, u_mesh = self.__heat_equation_model(self.get_inputs_asarray())
        self.local_data["u_mesh"] = u_mesh
        self.local_data["u"] = array([u])
