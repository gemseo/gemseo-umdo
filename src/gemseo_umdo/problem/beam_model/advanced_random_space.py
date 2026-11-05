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

from math import radians
from typing import TYPE_CHECKING
from typing import Final

from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.uncertainty.distribution import OTUniformDistribution_Settings
from gemseo.util.string import pretty_repr

from gemseo_umdo.problem.beam_model.core.variable import E
from gemseo_umdo.problem.beam_model.core.variable import L
from gemseo_umdo.problem.beam_model.core.variable import Rd
from gemseo_umdo.problem.beam_model.core.variable import Ry
from gemseo_umdo.problem.beam_model.core.variable import alpha
from gemseo_umdo.problem.beam_model.core.variable import b
from gemseo_umdo.problem.beam_model.core.variable import beta
from gemseo_umdo.problem.beam_model.core.variable import dy
from gemseo_umdo.problem.beam_model.core.variable import dz
from gemseo_umdo.problem.beam_model.core.variable import h
from gemseo_umdo.problem.beam_model.core.variable import t

if TYPE_CHECKING:
    from collections.abc import Mapping


_default_relative_dispersion: Final[float] = 0.05
"""The default dispersion relative to a reference value."""

_default_angle_dispersion: Final[float] = radians(1.0)
"""The default dispersion of an angle, in radians."""


class AdvancedBeamRandomSpace(RandomSpace):
    """The advanced uncertain space for the beam use case.

    The angles `alpha` and `beta` are expressed in radians.
    """

    def __init__(
        self,
        nominal_values: Mapping[str, float] | None = None,
        **dispersions: float,
    ) -> None:
        r"""
        Args:
            nominal_values: The nominal values $\mu$ of some uncertain variables.
                For missing ones,
                use the default values of the variables available in
                `gemseo_umdo.problem.beam_model.core.variable`.
            **dispersions: The dispersions $\delta$
                around the nominal values $\mu$.
                For a missing dispersion,
                use $\delta=0.05\mu$ for the positive variables
                (`b`, `h`, `t`, `L`, `E`, `Rd` and `Ry`),
                1 degree for the angles `alpha` and `beta`
                and 5% of the nominal values of `b` and `h`
                for the coordinates `dy` and `dz` respectively,
                whatever their nominal values.
                For the uniform variables `b`, `h`, `t` and `L`,
                the distribution is defined over $[\mu-\delta,\mu+\delta]$.
                For the other ones,
                the distribution is a normal one with standard deviation
                $\sigma=\delta/3$ truncated to $[\mu-3\sigma,\mu+3\sigma]$.

        Raises:
            ValueError: When a name in `nominal_values` or `dispersions`
                is not the name of an uncertain variable,
                when a dispersion is not positive,
                when the nominal value of a positive variable is not positive,
                or when the lower bound $\mu-\delta$
                of a positive variable
                (`b`, `h`, `t`, `L`, `E`, `Rd` or `Ry`)
                is not positive.
        """  # noqa: D205 D212 D415
        super().__init__()
        uniform_variables = (b, h, t, L)
        variables = (*uniform_variables, alpha, beta, dy, dz, E, Rd, Ry)
        uniform_variable_names = {variable.name for variable in uniform_variables}
        positive_variables = (*uniform_variables, E, Rd, Ry)
        positive_variable_names = {variable.name for variable in positive_variables}
        variable_names = [variable.name for variable in variables]
        nominal_values = nominal_values or {}
        unknown_names = (
            set(nominal_values).union(dispersions).difference(variable_names)
        )
        if unknown_names:
            msg = (
                "The following names are not uncertain variables: "
                f"{pretty_repr(unknown_names)}; "
                f"the available ones are {pretty_repr(variable_names, sort=False)}."
            )
            raise ValueError(msg)

        mus = {
            variable.name: nominal_values.get(variable.name, variable.value)
            for variable in variables
        }
        for variable in positive_variables:
            name = variable.name
            if mus[name] <= 0:
                msg = (
                    f"The nominal value {mus[name]} "
                    f"of the positive variable {name!r} must be positive."
                )
                raise ValueError(msg)

        signed_variable_default_dispersions = {
            alpha.name: _default_angle_dispersion,
            beta.name: _default_angle_dispersion,
            dy.name: _default_relative_dispersion * mus[b.name],
            dz.name: _default_relative_dispersion * mus[h.name],
        }
        for name, mu in mus.items():
            if name in dispersions:
                delta = dispersions[name]
                if delta <= 0:
                    msg = f"The dispersion {delta} of {name!r} must be positive."
                    raise ValueError(msg)
            elif name in positive_variable_names:
                delta = _default_relative_dispersion * mu
            else:
                delta = signed_variable_default_dispersions[name]

            lower, upper = mu - delta, mu + delta
            if name in positive_variable_names and lower <= 0:
                msg = (
                    f"The lower bound mu-delta={lower} "
                    f"of the positive variable {name!r} must be positive."
                )
                raise ValueError(msg)

            if name in uniform_variable_names:
                settings = OTUniformDistribution_Settings(minimum=lower, maximum=upper)
            else:
                settings = OTNormalDistribution_Settings(
                    mu=mu, sigma=delta / 3.0, lower_bound=lower, upper_bound=upper
                )

            self.add_variable(name, settings)
