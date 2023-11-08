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
"""The design space for the beam use case."""

from __future__ import annotations

from gemseo.algos.design_space import DesignSpace

from gemseo_umdo.use_cases.beam_model.core.design_space import BeamDesignVariables


class BeamDesignSpace(DesignSpace):
    """The design space for the beam use case."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        for variable in BeamDesignVariables:
            self.add_variable(
                variable.value.name,
                l_b=variable.value.l_b,
                u_b=variable.value.u_b,
                value=variable.value.value,
            )
