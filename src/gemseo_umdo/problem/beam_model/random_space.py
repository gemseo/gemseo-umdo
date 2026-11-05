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
"""The uncertain space for the beam use case."""

from __future__ import annotations

from typing import Final

from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.uncertainty.distribution import OTUniformDistribution_Settings

from gemseo_umdo.problem.beam_model.core.variable import E
from gemseo_umdo.problem.beam_model.core.variable import F
from gemseo_umdo.problem.beam_model.core.variable import sigma_all

_default_delta: Final[dict[str, float]] = {
    F.name: 10.0,
    E.name: 5.0,
    sigma_all.name: 5.0,
}
"""The default percentages of variation of the uncertain variables."""


class BeamRandomSpace(RandomSpace):
    r"""The uncertain space for the beam use case.

    It models the uncertainties about the load,
    the Young's modulus and the allowable stress;
    see [AdvancedBeamRandomSpace][gemseo_umdo.problem.beam_model.advanced_random_space.AdvancedBeamRandomSpace]
    for the additional uncertainties about the geometry,
    the other material properties
    and the position and direction of the load.

    $F$, $E$ and $\sigma_{\text{all}}$ are random variables
    with nominal values -200000, 73500 and 300
    and deviation values 10%, 5% and 5%.

    Their probability distribution are centered in these values
    denoted $\mu_F$, $\mu_E$ and $\mu_{\sigma_{\text{all}}}$.

    Precisely,
    a uniform distribution is defined
    by the minimum $\mu (1 - \delta)$ and the maximum $\mu (1 + \delta)$
    and a Gaussian distribution is defined
    by the mean $\mu$ and the standard deviation $|\mu|\delta/3$,
    where $\delta$ is an aforementioned deviation value.
    """  # noqa: E501

    def __init__(self, uniform: bool = True, **deltas: float) -> None:
        r"""
        Args:
            uniform: If `True`, use uniform distributions;
                otherwise, use Gaussian ones.
            **deltas: The percentage variations $\delta$ around the nominal values
                of the random variables.
        """  # noqa: D205 D212 D415
        super().__init__()
        for variable in [F, E, sigma_all]:
            nominal = variable.value
            name = variable.name
            delta = deltas.pop(name, _default_delta[name]) / 100
            if uniform:
                minimum, maximum = sorted([
                    nominal * (1 - delta),
                    nominal * (1 + delta),
                ])
                self.add_variable(
                    name,
                    OTUniformDistribution_Settings(minimum=minimum, maximum=maximum),
                )
            else:
                self.add_variable(
                    name,
                    OTNormalDistribution_Settings(
                        mu=nominal, sigma=abs(nominal) * delta / 3
                    ),
                )
