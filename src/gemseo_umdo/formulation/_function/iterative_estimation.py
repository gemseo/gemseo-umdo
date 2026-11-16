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
"""A function updating the estimation of a statistic and its Jacobian."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import atleast_1d
from numpy import atleast_2d

if TYPE_CHECKING:
    from gemseo.optimization.problem import EvaluationType
    from gemseo.util.typing import RealArray

    from gemseo_umdo.formulation._estimator.iterative_sampling.base import (
        BaseIterativeSamplingEstimator,
    )


class IterativeEstimation:
    """A functor to estimate a statistic and its Jacobian iteratively.

    Call this functor to update the estimation of the statistic
    and, if the Jacobians are estimated
    and the estimator can estimate the Jacobian of the statistic,
    of its Jacobian;
    then access the last estimations
    with `last_estimation` and `last_jacobian_estimation`
    and reset them with `reset()` before a new sampling.

    The [Sampling][gemseo_umdo.formulation.sampling.Sampling] U-MDO formulation
    passes such functors to a
    [DOELibrary][gemseo.doe.core.base_doe_library.BaseDOELibrary]
    as callback functions
    to update the statistics of the objective, constraints and observables.
    """

    statistic_estimator: BaseIterativeSamplingEstimator
    """The function to update the estimation of the statistic."""

    output_name: str
    """The name of the output."""

    output_statistic_name: str
    """The name of the statistic of the output."""

    output_statistic_jac_name: str
    """The name of the Jacobian of the statistic of the output."""

    last_estimation: RealArray
    """The last estimation of the statistic."""

    last_jacobian_estimation: RealArray | None
    """The last estimation of the Jacobian of the statistic.

    `None` when the Jacobian has not been estimated since the last reset.
    """

    n_samples: int
    """The number of samples used by the last estimations.

    It is the number of samples at which the output is available since the last reset
    and, when the Jacobians are estimated, its gradient too.
    """

    __estimate_jacobian: bool
    """Whether the Jacobians are estimated since the last reset."""

    def __init__(
        self,
        output_name: str,
        output_statistic_name: str,
        statistic_estimator: BaseIterativeSamplingEstimator,
        output_statistic_jac_name: str,
    ) -> None:
        """
        Args:
            output_name: The name of the output for which to estimate the statistic.
            output_statistic_name: The name of the statistic of the output.
            statistic_estimator: The function to update the estimation of the statistic.
            output_statistic_jac_name: The name of the Jacobian
                of the statistic of the output.
        """  # noqa: D202 D205 D212 D415
        self.output_name = output_name
        self.output_statistic_name = output_statistic_name
        self.output_statistic_jac_name = output_statistic_jac_name
        self.statistic_estimator = statistic_estimator
        self.reset()

    def __call__(self, index: int, evaluation: EvaluationType) -> None:
        """
        Args:
            index: The index of the evaluation.
            evaluation: The values of the function outputs
                and the values of their Jacobian.
        """  # noqa: D205, D212
        output_name = self.output_name
        value = evaluation[0].get(output_name)
        jac_value = evaluation[1].get(output_name)
        # A sample at which the output is missing,
        # or its gradient when the Jacobians are estimated,
        # is not used to estimate the statistic and its Jacobian,
        # so that both are estimated from the same samples.
        estimate_jacobian = self.__estimate_jacobian
        if value is None or (estimate_jacobian and jac_value is None):
            return

        value = atleast_1d(value)
        estimator = self.statistic_estimator
        # The estimation of the statistic must be updated
        # before the estimation of its Jacobian, which may use it.
        self.last_estimation = estimator.estimate_statistic(value)
        self.n_samples += 1
        if estimate_jacobian and estimator.can_compute_jacobian:
            self.last_jacobian_estimation = estimator.compute_jacobian(
                value, atleast_2d(jac_value)
            )

    def reset(self, estimate_jacobian: bool = False) -> None:
        """Reset the estimator and the last estimations.

        Args:
            estimate_jacobian: Whether the next samples are used
                to estimate the Jacobian of the statistic too.
        """
        # The estimator may have updated part of its state before failing,
        # so it is reset even if it has not returned any estimation.
        self.statistic_estimator.reset()
        self.last_estimation = array([])
        self.last_jacobian_estimation = None
        self.n_samples = 0
        self.__estimate_jacobian = estimate_jacobian
