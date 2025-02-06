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
from numpy import atleast_2d

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
    passes such functors to a
    [DOELibrary][gemseo.algos.doe.base_doe_library.BaseDOELibrary]
    as callback functions
    to update the statistics of the objective, constraints and observables.
    """

    statistic_estimator: BaseSamplingEstimator
    """The function to update the estimation of the statistic."""

    output_name: str
    """The name of the output."""

    output_statistic_name: str
    """The name of the statistic of the output."""

    last_estimation: RealArray
    """The last estimation of the statistic."""

    return_statistic_jacobian: bool
    """Whether the functor returns the Jacobian of the statistic estimation."""

    def __init__(
        self,
        output_name: str,
        output_statistic_name: str,
        statistic_estimator: BaseSamplingEstimator,
        return_statistic_jacobian: bool = False,
    ) -> None:
        """
        Args:
            output_name: The name of the output for which to estimate the statistic.
            output_statistic_name: The name of the statistic of the output.
            statistic_estimator: The function to update the estimation of the statistic.
            return_statistic_jacobian: Whether to return
                the Jacobian of the statistic estimation.
        """  # noqa: D202 D205 D212 D415
        self.output_name = output_name
        self.output_statistic_name = output_statistic_name
        self.last_estimation = array([])
        self.statistic_estimator = statistic_estimator
        self.return_statistic_jacobian = return_statistic_jacobian

    def __call__(self, index: int, evaluation: EvaluationType) -> RealArray:
        """
        Args:
            index: The index of the evaluation.
            evaluation: The values of the function outputs
                and the values of their Jacobian.

        Returns:
            The new estimation of the statistic or its Jacobian.
        """  # noqa: D205, D212
        value = atleast_1d(evaluation[0].get(self.output_name))
        if self.return_statistic_jacobian:
            jac_value = evaluation[1].get(self.output_name)
            self.last_estimation = self.statistic_estimator.compute_jacobian(
                value, atleast_2d(jac_value)
            )
        else:
            self.last_estimation = self.statistic_estimator.estimate_statistic(value)

        return self.last_estimation
