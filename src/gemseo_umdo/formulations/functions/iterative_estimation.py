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
from typing import Any

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import atleast_1d
from numpy import ndarray

if TYPE_CHECKING:
    from gemseo_umdo.formulations.statistics.iterative_sampling.sampling_estimator import (  # noqa: E501
        SamplingEstimator as IterativeSamplingEstimator,
    )


class IterativeEstimation(MDOFunction):
    """A function updating the estimation of a statistic."""

    _function: MDOFunction
    """The function computing the output for which we want to estimate the statistic."""

    _iterative_estimator: IterativeSamplingEstimator
    """The iterative statistic estimator."""

    _parameters: Any
    """The parameters of the iterative statistic estimator."""

    def __init__(
        self,
        name: str,
        function: MDOFunction,
        iterative_estimator: IterativeSamplingEstimator,
        **parameters: Any,
    ) -> None:
        """
        Args:
            name: The name of this iterative estimation function.
            function: The function computing the output
                for which we want to estimate the statistic.
            iterative_estimator: The iterative statistic estimator.
            **parameters: The parameters of the iterative statistic estimator.
        """  # noqa: D202 D205 D212 D415
        self._parameters = parameters
        self._function = function
        self._iterative_estimator = iterative_estimator
        super().__init__(self._estimate_statistic, name)

    def _estimate_statistic(self, input_value: ndarray) -> ndarray:
        """The function to be called.

        Args:
            input_value: The input value of the function.

        Returns:
            The new estimation of the statistic.
        """
        return self._iterative_estimator(
            atleast_1d(self._function.last_eval), **self._parameters
        )
