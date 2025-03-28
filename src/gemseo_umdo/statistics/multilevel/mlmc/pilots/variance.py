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
"""The variance-based pilot for the MLMC algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from numpy import array
from numpy import nanmean
from numpy import nansum
from numpy import nanvar

from gemseo_umdo.statistics.multilevel.mlmc.pilots.base_mlmc_pilot import BaseMLMCPilot

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy.typing import NDArray


class Variance(BaseMLMCPilot):
    """The variance-based pilot for the MLMC algorithm."""

    __delta: list[NDArray[float]]
    r"""The samples of $Y_0-Y_{-1},Y_2-Y_1,\ldots,Y_L-Y_{L-1}$.

    Namely,
    $(Y_\ell^{(\ell,n_\ell)}-Y_{\ell-1}^{(\ell,n_\ell)})_{0\leq \ell \leq L}$.
    """

    __sigma: list[NDArray[float]]
    r"""The samples of $Y_0+Y_{-1},Y_2+Y_1,\ldots,Y_L+Y_{L-1}$.

    Namely,
    $(Y_\ell^{(\ell,n_\ell)}+Y_{\ell-1}^{(\ell,n_\ell)})_{0\leq \ell \leq L}$.
    """

    def __init__(  # noqa: D107
        self, sampling_ratios: NDArray[float], costs: NDArray[float]
    ) -> None:
        super().__init__(sampling_ratios, costs)
        n_levels = len(sampling_ratios)
        self.__delta = [array([]) for _ in range(n_levels)]
        self.__sigma = [array([]) for _ in range(n_levels)]

    def _compute_statistic(self) -> float:  # noqa: D102
        # El Amri et al., Eq. 29, Multilevel Surrogate-based Control Variates, 2023.
        # Var_MLMC[Y] = sum_l Var^l[Y_l] - Var^l[Y_{l-1}]
        #             = sum_l Var^l[(D_l+S_l)/2] - Var^l[(S_l+D_l)/2]
        #               where D_l = Y_l-Y_{l-1} and S_l = Y_l+Y_{l-1}
        return nansum([
            nanvar((delta + sigma) / 2) - nanvar((sigma - delta) / 2)
            for delta, sigma in zip(self.__delta, self.__sigma)
        ])

    def _compute_V_l(  # noqa: D102 N802
        self,
        levels: Iterable[int],
        samples: Sequence[NDArray[float]],
        *pilot_parameters: Any,
    ) -> NDArray[float]:
        # Mycek and De Lozzo, Table 1,
        # Multilevel Monte Carlo covariance estimation
        # for the computation of Sobol' indices, 2019.
        # V_l = (M4[D_l]M4[S_l])**0.5
        for level in levels:
            samples_ = samples[level]
            self.__delta[level] = samples_[:, 0] - samples_[:, 1]
            self.__sigma[level] = samples_.sum(1)

        return array([
            (
                nanmean((delta - delta.mean()) ** 4)
                * nanmean((sigma - sigma.mean()) ** 4)
            )
            ** 0.5
            for delta, sigma in zip(self.__delta, self.__sigma)
        ])
