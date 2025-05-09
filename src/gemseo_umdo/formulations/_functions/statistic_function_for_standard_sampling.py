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
"""A function to compute a statistic from `Sampling`.

See also [Sampling][gemseo_umdo.formulations.sampling.Sampling].
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

from numpy import newaxis

from gemseo_umdo.formulations._functions.base_statistic_function_for_sampling import (
    BaseStatisticFunctionForSampling,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.sampling import Sampling

SamplingT = TypeVar("SamplingT", bound="Sampling")


class StatisticFunctionForStandardSampling(BaseStatisticFunctionForSampling[SamplingT]):
    """A function to compute a statistic from `Sampling`."""

    def _compute_statistic_estimation(
        self, output_data: dict[str, RealArray]
    ) -> RealArray:
        return self._statistic_estimator.estimate_statistic(
            output_data[self._output_name]
        )

    def _compute_statistic_jacobian_estimation(
        self, data: dict[str, RealArray]
    ) -> RealArray:
        return self._statistic_estimator.compute_jacobian(
            data[self._output_name], data[self._output_jac_name]
        )

    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, Any]:
        formulation = self._umdo_formulation
        problem = formulation.mdo_formulation.optimization_problem
        database = problem.database
        formulation.compute_samples(
            problem, input_data, compute_jacobian=estimate_jacobian
        )
        data = {}
        for output_name in database.get_function_names(skip_grad=False):
            history = database.get_function_history(output_name)
            ndim = history.ndim
            if output_name.startswith(database.GRAD_TAG):
                if ndim == 2:
                    history = history[:, newaxis, :]
                elif ndim == 1:
                    history = history[:, newaxis, newaxis]
            elif ndim == 1:
                history = history[:, newaxis]

            data[output_name] = history

        return data
