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
"""The discipline for the beam use case."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import fields
from typing import TYPE_CHECKING

from gemseo.discipline import Discipline
from numpy import array

from gemseo_umdo.use_cases.beam_model.core.model import BeamModel
from gemseo_umdo.use_cases.beam_model.core.output_data import BeamModelOutputData
from gemseo_umdo.use_cases.beam_model.core.variables import E
from gemseo_umdo.use_cases.beam_model.core.variables import F
from gemseo_umdo.use_cases.beam_model.core.variables import L
from gemseo_umdo.use_cases.beam_model.core.variables import alpha
from gemseo_umdo.use_cases.beam_model.core.variables import b
from gemseo_umdo.use_cases.beam_model.core.variables import beta
from gemseo_umdo.use_cases.beam_model.core.variables import dy
from gemseo_umdo.use_cases.beam_model.core.variables import dz
from gemseo_umdo.use_cases.beam_model.core.variables import h
from gemseo_umdo.use_cases.beam_model.core.variables import nu
from gemseo_umdo.use_cases.beam_model.core.variables import rho
from gemseo_umdo.use_cases.beam_model.core.variables import t

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping


class Beam(Discipline):
    """The beam discipline.

    See Also:
        [BeamModel][gemseo_umdo.use_cases.beam_model.core.model.BeamModel]
        for more information about the beam model.
    """

    def __init__(self, n_y: int = 3, n_z: int = 3) -> None:
        """
        Args:
            n_y: The number of discretization points in the y-direction.
            n_z: The number of discretization points in the z-direction.
        """  # noqa: D205 D212 D415
        super().__init__()
        input_variables = [b, h, t, L, E, alpha, beta, dy, dz, rho, F, nu]
        self.input_grammar.update_from_names([
            variable.name for variable in input_variables
        ])
        self.output_grammar.update_from_names([
            f.name for f in fields(BeamModelOutputData)
        ])
        self.default_input_data = {
            variable.name: array([variable.value]) for variable in input_variables
        }
        self.__beam_model = BeamModel(n_y, n_z)

    def _run(self, input_data: StrKeyMapping) -> None:
        input_data = {key: val[0] for key, val in self.get_input_data().items()}
        for name, value in asdict(self.__beam_model(**input_data)).items():
            # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data, get(name) or get_merged_data()  # noqa: E501
            self.io.data[name] = value.ravel()
