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

from typing import TYPE_CHECKING
from typing import Final

from gemseo.discipline import Discipline
from numpy import array

from gemseo_umdo.use_cases.beam_model.core.variables import sigma_all

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping


class BeamConstraints(Discipline):
    r"""The discipline computing the constraints of the beam problem.

    More particularly,
    the left-hand sides of

    - the stress constraints $\sigma_{\mathrm{VM}}/\sigma_{\mathrm{all}} \leq 1$,
    - the displacements constraints $\Delta/\Delta_{\mathrm{min}} \geq 1$.
    """

    __c_stress: Final[str] = "c_stress"
    __sigma_vm: Final[str] = "sigma_vm"
    __c_displ: Final[str] = "c_displ"
    __displ: Final[str] = "displ"

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.input_grammar.update_from_names([
            self.__displ,
            self.__sigma_vm,
            sigma_all.name,
        ])
        self.output_grammar.update_from_names([self.__c_displ, self.__c_stress])
        self.default_input_data = {
            sigma_all.name: array([sigma_all.value]),
            self.__sigma_vm: array([300.0]),
            self.__displ: array([100.0]),
        }

    def _run(self, input_data: StrKeyMapping) -> None:
        self.io.output_data[self.__c_stress] = (
            self.io.input_data[self.__sigma_vm] / self.io.input_data[sigma_all.name]
        )
        self.io.output_data[self.__c_displ] = self.io.input_data[self.__displ] / 100.0
