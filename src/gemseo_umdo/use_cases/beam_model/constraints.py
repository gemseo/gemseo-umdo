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
"""The discipline computing the constraints for the beam use case."""

from __future__ import annotations

from typing import Final

from gemseo.core.discipline import MDODiscipline
from numpy import array

from gemseo_umdo.use_cases.beam_model.core.variables import sigma_all


class BeamConstraints(MDODiscipline):
    r"""The discipline computing the constraints of the beam problem.

    - Stress constraints: $\sigma_{\mathrm{all}}/(\sigma_{\mathrm{VM}}+1)$.
    - Displacements constraints: $\Delta_{\mathrm{min}}/(\Delta+0.1)$.
    """

    __C_STRESS: Final[str] = "c_stress"
    __SIGMA_VM: Final[str] = "sigma_vm"
    __C_DISPL: Final[str] = "c_displ"
    __DISPL: Final[str] = "displ"

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.input_grammar.update_from_names([
            self.__DISPL,
            self.__SIGMA_VM,
            sigma_all.name,
        ])
        self.output_grammar.update_from_names([self.__C_DISPL, self.__C_STRESS])
        self.default_inputs = {
            sigma_all.name: array([sigma_all.value]),
            self.__SIGMA_VM: array([300.0]),
            self.__DISPL: array([100.0]),
        }

    def _run(self) -> None:
        self._local_data[self.__C_STRESS] = self._local_data[sigma_all.name] / (
            self._local_data[self.__SIGMA_VM] + 1.0
        )
        self._local_data[self.__C_DISPL] = 100.0 / (
            self._local_data[self.__DISPL] + 0.1
        )
