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
r"""A level $\ell$ for the MLMC-MLCV algorithm."""

from __future__ import annotations

from dataclasses import dataclass

from gemseo.core.mdo_functions.mdo_function import MDOFunction


@dataclass
class Level:
    r"""A level $\ell$ for the MLMC-MLCV algorithm."""

    model: MDOFunction
    r"""The model $f_\ell$ to sample.

    This model can be set from any callable taking a NumPy array of float numbers as
    input and outputting either a float number or a NumPy array of float numbers.
    """

    surrogate_model: tuple[MDOFunction, float]
    r"""The surrogate model $g_\ell$ approximating $f_\ell$.

    More precisely,
    $g_\ell$ and its statistic for the
    [MLMCMLCV][gemseo_umdo.statistics.multilevel.mlmc_mlcv.mlmc_mlcv.MLMCMLCV]
    algorithm.

    The surrogate model can be set from any callable
    taking a NumPy array of float numbers as input
    and outputting either a float number or a NumPy array of float numbers.
    """

    difference_surrogate_model: tuple[MDOFunction, float] = ()
    r"""The surrogate model $h_\ell$ approximating $f_\ell-f_{\ell-1}$.

    More precisely,
    $h_\ell$ and its statistic for the
    [MLMCMLCV][gemseo_umdo.statistics.multilevel.mlmc_mlcv.mlmc_mlcv.MLMCMLCV]
    algorithm.

    Empty at level $\ell=0$.

    The surrogate model can be set from any callable
    taking a NumPy array of float numbers as input
    and outputting either a float number or a NumPy array of float numbers.
    """

    cost: float | None = None
    r"""The cost $\mathcal{C}_\ell$ to evaluate $f_\ell$, if known."""

    n_cost_estimation_samples: int = 1
    r"""The number of $f_\ell$ calls to estimate $\mathcal{C}_\ell$.

    It will be used only if `cost` is `None`.
    """

    n_initial_samples: int = 10
    r"""The number of samples $n_\ell$ at the first iteration of the algorithm."""

    sampling_ratio: float = 2.0
    r"""The factor $r_\ell$ by which $n_\ell$ is increased."""

    def __post_init__(self) -> None:
        if self.difference_surrogate_model:
            self.difference_surrogate_model = (
                MDOFunction(self.difference_surrogate_model[0], "h"),
                self.difference_surrogate_model[1],
            )
        self.model = MDOFunction(self.model, "f")
        self.surrogate_model = (
            MDOFunction(self.surrogate_model[0], "g"),
            self.surrogate_model[1],
        )
