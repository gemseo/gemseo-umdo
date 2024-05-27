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
"""A function updating the estimation of a statistic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import atleast_1d

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import EvaluationType
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations._statistics.iterative_sampling.base_sampling_estimator import (  # noqa: E501
        BaseSamplingEstimator,
    )


class IterativeEstimation:
    """A functor to estimate a statistic iteratively.

    Call this functor to update the estimation of the statistic
    and access the last evaluation with :attr:`.last_evaluation`.

    The [Sampling][gemseo_umdo.formulations.sampling.Sampling] U-MDO formulation
    passes such functors to a [DOELibrary][gemseo.algos.doe.doe_library.DOELibrary]
    as callback functions
    to update the statistics of the objective, constraints and observables.
    """

    _update_estimation: BaseSamplingEstimator
    """The function to update the estimation of the statistic."""

    _name: str
    """The name of the output."""

    _last_estimation: RealArray
    """The last estimation of the statistic."""

    def __init__(
        self,
        name: str,
        update_estimation: BaseSamplingEstimator,
    ) -> None:
        """
        Args:
            name: The name of the output for which to estimate the statistic.
            update_estimation: The function to update the estimation of the statistic.
        """  # noqa: D202 D205 D212 D415
        self._name = name
        self._last_estimation = array([])
        self._update_estimation = update_estimation

    def __call__(self, index: int, evaluation: EvaluationType) -> RealArray:
        """
        Args:
            index: The index of the evaluation.
            evaluation: The values of the function outputs
                and the values of their Jacobian.

        Returns:
            The new estimation of the statistic.
        """  # noqa: D205, D212
        self._last_estimation = self._update_estimation(
            atleast_1d(evaluation[0][self._name])
        )
        return self._last_estimation

    @property
    def last_estimation(self) -> RealArray:
        """The last estimation of the statistic."""
        return self._last_estimation
