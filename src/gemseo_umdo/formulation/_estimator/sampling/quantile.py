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
from decimal import Decimal
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from numpy import arange
from numpy import array
from numpy import quantile

from gemseo_umdo.formulation._estimator.sampling.base import BaseSamplingEstimator

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from gemseo.util.typing import RealArray

logger = logging.getLogger(__name__)

_min_tail_size: Final[int] = 10
"""The minimum number of samples expected in the tail beyond the quantile."""

_level_name: Final[str] = "level"
"""The name of the parameter for the level of the quantile."""


def _check_level(level: float) -> None:
    """Check that the level of a quantile is in $]0,1[$.

    Args:
        level: The level of the quantile.

    Raises:
        ValueError: When the level is not in $]0,1[$.
    """
    if not 0 < level < 1:
        msg = f"The level of the quantile must be in ]0,1[; got {level}."
        raise ValueError(msg)


class Quantile(BaseSamplingEstimator):
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

    __level: float
    """The level of the quantile, in $]0,1[$."""

    __method: str
    """The name of the method of [numpy.quantile][numpy.quantile]."""

    __is_size_checked: bool
    """Whether the number of samples has been checked against the level."""

    default_level: ClassVar[float] = 0.95
    """The default level of the quantile."""

    def __init__(self, level: float = default_level, method: str = "linear") -> None:
        r"""
        Args:
            level: The level $\alpha$ of the quantile, in $]0,1[$.
            method: The name of the method of [numpy.quantile][numpy.quantile]
                to estimate the quantile.

        Raises:
            ValueError: When the level is not in $]0,1[$
                or when the method is not a method of
                [numpy.quantile][numpy.quantile].
        """  # noqa: D205 D212 D415
        _check_level(level)
        # The method is checked with a cheap call to numpy.quantile
        # so that a wrong method fails before any sampling,
        # as NumPy does not expose its list of methods.
        try:
            quantile(array([0.0, 1.0]), level, method=method)
        except ValueError as error:
            msg = (
                "The method of the quantile must be a method of numpy.quantile; "
                f"got '{method}'."
            )
            raise ValueError(msg) from error

        super().__init__()
        self.__level = level
        self.__method = method
        self.__is_size_checked = False

    @classmethod
    def update_parameters(
        cls, parameters: MutableMapping[str, Any], conservative_from_below: bool | None
    ) -> MutableMapping[str, Any]:
        """
        Raises:
            ValueError: When the level is not in $]0,1[$.
        """  # noqa: D205 D212 D415
        # The default level is filled in so that the name of the statistic includes it.
        level = parameters.setdefault(_level_name, cls.default_level)
        # The level is checked before its conservative update
        # so that an error reports the level passed by the user.
        _check_level(level)
        if conservative_from_below is not None:
            # The conversion to float handles NumPy scalars,
            # whose repr is not a valid decimal string, e.g. np.float64(0.9).
            level = float(level)
            # The decimal arithmetic avoids names like Q[f; 0.09999999999999998]
            # for the complementary level of 0.9.
            complementary_level = float(1 - Decimal(repr(level)))
            parameters[_level_name] = (
                min(level, complementary_level)
                if conservative_from_below
                else max(level, complementary_level)
            )

        return parameters

    def estimate_statistic(self, samples: RealArray) -> RealArray:
        self.__check_size(len(samples))
        return quantile(samples, self.__level, axis=0, method=self.__method)

    def compute_jacobian(self, samples: RealArray, jac_samples: RealArray) -> RealArray:
        n_samples = len(samples)
        # The quantile of the sorted indices 0, ..., N-1 is the position j + w
        # of the quantile in the order statistics.
        position = quantile(arange(n_samples), self.__level, method=self.__method)
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
        tail_size = n_samples * min(self.__level, 1 - self.__level)
        if tail_size < _min_tail_size:
            logger.warning(
                "The estimation of the %s-quantile from %s samples may be inaccurate "
                "as the expected number of samples beyond the quantile is %.3g < %s.",
                self.__level,
                n_samples,
                tail_size,
                _min_tail_size,
            )
