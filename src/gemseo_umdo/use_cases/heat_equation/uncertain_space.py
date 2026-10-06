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
"""The uncertain space to be used with the heat equation discipline."""

from __future__ import annotations

from gemseo.space import RandomSpace
from numpy import pi
from gemseo.uncertainty.distribution import OTUniformDistribution_Settings


class HeatEquationUncertainSpace(RandomSpace):
    """The uncertain space to be used with the heat equation discipline."""

    def __init__(self, nu_bounds: tuple[float, float] = (0.001, 0.009)) -> None:
        r"""
        Args:
            nu_bounds: The lower and upper bounds
                of the thermal diffusivity $\nu$.
        """  # noqa: D205 D212 D415
        distribution_name = "OTUniformDistribution"
        super().__init__()
        self.add_variable("X_1", OTUniformDistribution_Settings(minimum=-pi, maximum=pi))
        self.add_variable("X_2", OTUniformDistribution_Settings(minimum=-pi, maximum=pi))
        self.add_variable("X_3", OTUniformDistribution_Settings(minimum=-pi, maximum=pi))
        self.add_variable(
            "X_4", OTUniformDistribution_Settings(minimum=nu_bounds[0], maximum=nu_bounds[1]
        ))
        self.add_variable("X_5", OTUniformDistribution_Settings(minimum=-1.0, maximum=1.0))
        self.add_variable("X_6", OTUniformDistribution_Settings(minimum=-1.0, maximum=1.0))
        self.add_variable("X_7", OTUniformDistribution_Settings(minimum=-1.0, maximum=1.0))
