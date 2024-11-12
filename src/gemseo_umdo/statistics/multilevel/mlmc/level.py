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
r"""A level $\ell$ for the MLMC algorithm."""

from __future__ import annotations

from dataclasses import dataclass

from gemseo.core.mdo_functions.mdo_function import MDOFunction


@dataclass
class Level:
    r"""A level $\ell$ for the MLMC algorithm."""

    model: MDOFunction
    r"""The model $f_\ell$ to sample.

    This model can be set from any callable taking a NumPy array of float numbers as
    input and outputting either a float number or a NumPy array of float numbers.
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
    r"""The number $r_\ell$ by which $n_\ell$ is increased."""

    def __post_init__(self) -> None:
        self.model = MDOFunction(self.model, "f")
