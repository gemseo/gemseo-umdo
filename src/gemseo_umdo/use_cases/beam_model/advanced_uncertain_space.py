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
"""The advanced uncertain space for the beam use case."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.algos.parameter_space import ParameterSpace

from gemseo_umdo.use_cases.beam_model.core.variables import E
from gemseo_umdo.use_cases.beam_model.core.variables import L
from gemseo_umdo.use_cases.beam_model.core.variables import Variable
from gemseo_umdo.use_cases.beam_model.core.variables import alpha
from gemseo_umdo.use_cases.beam_model.core.variables import b
from gemseo_umdo.use_cases.beam_model.core.variables import beta
from gemseo_umdo.use_cases.beam_model.core.variables import dy
from gemseo_umdo.use_cases.beam_model.core.variables import dz
from gemseo_umdo.use_cases.beam_model.core.variables import h
from gemseo_umdo.use_cases.beam_model.core.variables import t

if TYPE_CHECKING:
    from collections.abc import Mapping


class AdvancedBeamUncertainSpace(ParameterSpace):
    """The advanced uncertain space for the beam use case."""

    __DEFAULT_DISPERSION: ClassVar[float] = 1.0

    def __init__(
        self, nominal_values: Mapping[str, float] | None = None, **dispersions: float
    ) -> None:
        """
        Args:
            nominal_values: The nominal values of some uncertain variables.
                For missing ones,
                use the default values of the variables available in
                ``gemseo_umdo.use_cases.beam_model.core.variables``.
            **dispersions: The dispersions around the nominal values.
        """  # noqa: D205 D212 D415
        super().__init__()
        variables = [
            b,
            h,
            t,
            L,
            alpha,
            beta,
            dy,
            dz,
            E,
            Variable("Rd", 180.0),
            Variable("Ry", 600.0),
        ]
        self.__nominal_values = {
            variable.name: variable.value for variable in variables
        }
        if nominal_values is not None:
            self.__nominal_values.update(nominal_values)

        for variable in variables[:4]:
            name = variable.name
            nominal_value = variable.value
            delta = dispersions.get(name, self.__DEFAULT_DISPERSION)
            self.add_random_variable(
                name,
                "OTUniformDistribution",
                minimum=nominal_value - delta,
                maximum=nominal_value + delta,
            )

        for variable in variables[4:]:
            self.__add_truncated_normal(variable.name, **dispersions)

    def __add_truncated_normal(self, name: str, **dispersions: float) -> None:
        r"""Add a truncated normal distribution to the parameter space.

        Use the ``.__nominal_values[name]`` as mean.

        Args:
            name: The name of the random variable.
            **dispersions: The dispersion $\delta$ around the mean $\mu$
                to define the standard deviation $\sigma=\delta/3$;
                the distribution is truncated
                to the interval $[\mu-3\sigma,\mu+3\sigma]$;
                if the dispersion is missing, use ``.__DEFAULT_DISPERSION``.
        """
        nominal_value = self.__nominal_values[name]
        sigma = dispersions.get(name, self.__DEFAULT_DISPERSION / 3.0)
        self.add_random_variable(
            name,
            "OTNormalDistribution",
            mu=nominal_value,
            sigma=sigma,
            lower_bound=nominal_value - 3 * sigma,
            upper_bound=nominal_value + 3 * sigma,
        )
