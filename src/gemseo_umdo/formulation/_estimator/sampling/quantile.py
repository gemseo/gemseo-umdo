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
"""Estimator of a quantile for sampling-based U-MDO formulations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Final

from numpy import arange
from numpy import quantile

from gemseo_umdo.formulation._estimator.sampling.base import BaseSamplingEstimator
from gemseo_umdo.statistic.settings import Quantile_Settings

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


logger = logging.getLogger(__name__)

_min_tail_size: Final[int] = 10
"""The minimum number of samples expected in the tail beyond the quantile."""


class Quantile(BaseSamplingEstimator[Quantile_Settings]):
    r"""Estimator of a quantile.

    The $\alpha$-quantile is estimated from the order statistics
    $\varphi_{(1)}\leq\ldots\leq\varphi_{(N)}$ of the samples
    with one of the methods of
    [numpy.quantile][numpy.quantile]
    [@hyndman1996sample],
    which all read $(1-w)\varphi_{(j)}+w\varphi_{(j+1)}$
    where $j$ and $w\in[0,1[$ depend on $\alpha$, $N$ and the method.

    The Jacobian is the same combination of the Jacobians of the samples
    $\varphi_{(j)}$ and $\varphi_{(j+1)}$,
    i.e. the pathwise estimator of the quantile sensitivity [@hong2009estimating].
    The estimator is only piecewise differentiable:
    its Jacobian jumps when the ranking of the samples changes,
    and one of the tied samples is used in case of ties.
    It is the exact Jacobian of the estimated quantile
    wherever the ranking of the samples does not change,
    but not a consistent estimator of the Jacobian of the quantile,
    as its variance does not decrease when the number of samples increases.
    """

    __is_size_checked: bool
    """Whether the number of samples has been checked against the level."""

    def __init__(self, settings: Quantile_Settings) -> None:
        """
        Args:
            settings: The settings of the quantile.
        """  # noqa: D205 D212 D415
        super().__init__(settings)
        self.__is_size_checked = False

    def estimate_statistic(self, samples: RealArray) -> RealArray:
        self.__check_size(len(samples))
        return quantile(
            samples, self._settings.level, axis=0, method=self._settings.method
        )

    def compute_jacobian(self, samples: RealArray, jac_samples: RealArray) -> RealArray:
        n_samples = len(samples)
        # The quantile of the sorted indices 0, ..., N-1 is the position j + w
        # of the quantile in the order statistics.
        position = quantile(
            arange(n_samples), self._settings.level, method=self._settings.method
        )
        index = int(position)
        weight = position - index
        # Only the order statistics j and j + 1 are needed, not a full sort.
        ranks = samples.argpartition((index, index + 1) if weight else index, axis=0)
        components = arange(samples.shape[1])
        jac = (1 - weight) * jac_samples[ranks[index], components]
        if weight:
            jac += weight * jac_samples[ranks[index + 1], components]

        return jac

    def __check_size(self, n_samples: int) -> None:
        """Log a warning once if the tail beyond the quantile has too few samples.

        Args:
            n_samples: The number of samples.
        """
        if self.__is_size_checked:
            return

        self.__is_size_checked = True
        tail_size = n_samples * min(self._settings.level, 1 - self._settings.level)
        if tail_size < _min_tail_size:
            logger.warning(
                "The estimation of the %s-quantile from %s samples may be inaccurate "
                "as the expected number of samples beyond the quantile is %.3g < %s.",
                self._settings.level,
                n_samples,
                tail_size,
                _min_tail_size,
            )
