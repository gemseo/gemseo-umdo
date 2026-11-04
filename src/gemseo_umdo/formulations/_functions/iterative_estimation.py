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

    from gemseo_umdo.formulations._statistics.iterative_sampling.base_sampling_estimator import (  # noqa: E501
        BaseSamplingEstimator,
    )


class IterativeEstimation:
    """A functor to estimate a statistic and its Jacobian iteratively.

    Call this functor to update the estimation of the statistic
    and, if the Jacobian of the output is evaluated
    and the estimator can estimate the Jacobian of the statistic,
    of its Jacobian;
    then access the last estimations
    with `last_estimation` and `last_jacobian_estimation`
    and reset them with `reset()` before a new sampling.

    The [Sampling][gemseo_umdo.formulations.sampling.Sampling] U-MDO formulation
    passes such functors to a
    [DOELibrary][gemseo.doe.core.base_doe_library.BaseDOELibrary]
    as callback functions
    to update the statistics of the objective, constraints and observables.
    """

    statistic_estimator: BaseSamplingEstimator
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
    """The number of samples used by the last estimation of the statistic.

    It is the number of samples at which the output is available since the last reset.
    """

    n_jacobian_samples: int
    """The number of samples used by the last estimation of the Jacobian.

    It is the number of samples
    at which both the output and its gradient are available since the last reset,
    so that it can be smaller than
    [n_samples][gemseo_umdo.formulations._functions.iterative_estimation.IterativeEstimation.n_samples].
    """

    def __init__(
        self,
        output_name: str,
        output_statistic_name: str,
        statistic_estimator: BaseSamplingEstimator,
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
        # A sample at which the output is missing,
        # e.g. because its evaluation failed,
        # is not used to estimate the statistic and its Jacobian.
        value = evaluation[0].get(output_name)
        if value is None:
            return

        value = atleast_1d(value)
        estimator = self.statistic_estimator
        # The estimation of the statistic must be updated
        # before the estimation of its Jacobian, which may use it.
        self.last_estimation = estimator.estimate_statistic(value)
        self.n_samples += 1
        # A sample at which the gradient of the output is missing
        # is not used to estimate the Jacobian of the statistic.
        jac_value = evaluation[1].get(output_name)
        if jac_value is not None and self.estimates_jacobian:
            self.last_jacobian_estimation = estimator.compute_jacobian(
                value, atleast_2d(jac_value)
            )
            self.n_jacobian_samples += 1

    @property
    def estimates_jacobian(self) -> bool:
        """Whether the Jacobian of the statistic is estimated when possible."""
        return self.statistic_estimator.can_compute_jacobian

    def reset(self) -> None:
        """Reset the estimator and the last estimations."""
        # The estimator may have updated part of its state before failing,
        # so it is reset even if it has not returned any estimation.
        self.statistic_estimator.reset()
        self.last_estimation = array([])
        self.last_jacobian_estimation = None
        self.n_samples = 0
        self.n_jacobian_samples = 0
