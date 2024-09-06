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
"""The base pilot for multilevel algorithms."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from numpy import argmax
from numpy import array
from numpy import nan

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy.typing import NDArray


class BasePilot(metaclass=ABCGoogleDocstringInheritanceMeta):
    r"""The base pilot for multilevel algorithms.

    A pilot is associated with a statistic, e.g. mean.
    The method
    [compute_next_level_and_statistic()][gemseo_umdo.statistics.multilevel.base_pilot.BasePilot.compute_next_level_and_statistic]
    returns a multilevel estimation of the statistic based on the current samples
    and the next level $\ell^*$ of the telescopic sum to sample
    in order to improve this estimation.

    This level $\ell^*$ maximizes the criterion

    $$\frac{\mathcal{V}_\ell}
    {r_\ell n_\ell^2(\mathcal{C}_\ell+\mathcal{C}_{\ell-1})}$$

    where $\mathcal{C}_{\ell}$ is the unit evaluation cost
    of the model $f_\ell$ (with $\mathcal{C}_{-1}=0$),
    $n_\ell$ is the current number of evaluations of $f_\ell$
    and $r_\ell$ is the factor by which $n_\ell$ would be increased
    by choosing the level $\ell$.
    Regarding $\mathcal{V}_\ell$,
    it represents the variance of the $\ell$-th term of the telescopic sum
    characteristic of the MLMC techniques.
    For instance,
    $\mathcal{V}_\ell=\mathbb{E}[Y_\ell-Y_{\ell}]$ in the case of the expectation.

    See Also:
        El Amri et al., Algo. 1, Multilevel Surrogate-based Control Variates, 2023.
    """

    V_l: NDArray[float]
    r"""The terms variances $\mathcal{V}_0,\ldots,\mathcal{V}_L$."""

    __costs: NDArray[float]
    r"""The unit sampling costs of each level of the telescopic sum.

    Namely,
    $(\mathcal{C}_{\ell-1}+\mathcal{C}_\ell)_{\ell\in\{0,\ldots,L\}}$
    with $\mathcal{C}_{-1}=0$.
    """

    __r_l: NDArray[float]
    r"""The sampling ratios of each level of the telescopic sum.

    Namely, $r_0,r_1,\ldots,r_L$.
    """

    def __init__(self, sampling_ratios: NDArray[float], costs: NDArray[float]) -> None:
        r"""
        Args:
            sampling_ratios: The sampling ratios $r_0,\ldots,r_L$;
                the sampling ratio $r_\ell$ is
                the factor by which $n_\ell$ is increased
                between two sampling steps on the level $ell$.
            costs: The unit sampling costs of each level of the telescopic sum.
                Namely,
                $(\mathcal{C}_{\ell-1}+\mathcal{C}_\ell)_{\ell\in\{0,\ldots,L\}}$
                with $\mathcal{C}_{-1}=0$.
        """  # noqa: D205 D212 D415
        self.__costs = costs
        self.__r_l = sampling_ratios
        self.V_l = array([nan] * len(self.__r_l))

    def compute_next_level_and_statistic(
        self,
        levels: Iterable[int],
        total_n_samples: NDArray[int],
        samples: Sequence[NDArray[float]],
        *pilot_parameters: Any,
    ) -> tuple[int, NDArray[float]]:
        r"""Compute the next level $\ell^*$ to sample and estimate the statistic.

        Args:
            levels: The levels that have just been sampled.
            total_n_samples: The total number of samples of each level.
            samples: The samples of the different quantities of each level.
            *pilot_parameters: The parameters of the pilot.

        Returns:
            The next level $\ell^*$ to sample and an estimation of the statistic.
        """
        self.V_l = self._compute_V_l(levels, samples, *pilot_parameters)
        # WARNING: do not replace "/ total_n_samples / total_n_samples"
        #          by "/ total_n_samples**2"
        #          to avoid numerical division issue due to an excessively big number.
        return (
            argmax(
                self.V_l / self.__r_l / total_n_samples / total_n_samples / self.__costs
            ),
            self._compute_statistic(),
        )

    @abstractmethod
    def _compute_statistic(self) -> float:
        """Estimate the statistic associated with this pilot."""

    @abstractmethod
    def _compute_V_l(  # noqa: N802
        self,
        levels: Iterable[int],
        samples: Sequence[NDArray[float]],
        *pilot_parameters: Any,
    ) -> NDArray[float]:
        r"""Compute the terms variances $\mathcal{V}_0,\ldots,\mathcal{V}_L$.

        Args:
            levels: The previous sampled levels.
            samples: The samples of the different quantities for each level.
            *pilot_parameters: The parameters of the pilot.

        Returns:
            The terms variances $\mathcal{V}_0,\ldots,\mathcal{V}_L$.
        """
