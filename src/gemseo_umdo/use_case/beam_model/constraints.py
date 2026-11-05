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

from gemseo_umdo.use_case.beam_model.core.variable import sigma_all

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping


_c_stress: Final[str] = "c_stress"
"""The name of the left-hand side of the stress constraints."""

_sigma_vm: Final[str] = "sigma_vm"
"""The name of the von Mises stress."""

_c_displ: Final[str] = "c_displ"
"""The name of the left-hand side of the displacement constraints."""

_displ: Final[str] = "displ"
"""The name of the displacement."""


class BeamConstraints(Discipline):
    r"""The discipline computing the constraints of the beam problem.

    More particularly,
    the left-hand sides of

    - the stress constraints $\sigma_{\mathrm{VM}}/\sigma_{\mathrm{all}} \leq 1$,
    - the displacements constraints $\Delta/\Delta_{\mathrm{min}} \geq 1$.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.input_grammar.update_from_names([
            _displ,
            _sigma_vm,
            sigma_all.name,
        ])
        self.output_grammar.update_from_names([_c_displ, _c_stress])
        self.default_input_data = {
            sigma_all.name: array([sigma_all.value]),
            _sigma_vm: array([300.0]),
            _displ: array([100.0]),
        }

    def _run(self, input_data: StrKeyMapping) -> None:
        self.io.output_data[_c_stress] = (
            self.io.input_data[_sigma_vm] / self.io.input_data[sigma_all.name]
        )
        self.io.output_data[_c_displ] = self.io.input_data[_displ] / 100.0
