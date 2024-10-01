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
from typing import TypeVar

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
            output_data[self._function_name]
        )

    def _compute_statistic_jacobian_estimation(
        self, output_data: dict[str, RealArray]
    ) -> RealArray:
        return self._statistic_estimator.compute_jacobian(
            output_data[self._function_name], output_data[self._function_jac_name]
        )

    def _compute_output_data(
        self,
        input_data: RealArray,
        output_data: dict[str, RealArray],
        compute_jacobian: bool = False,
    ) -> None:
        formulation = self._umdo_formulation
        problem = formulation.mdo_formulation.optimization_problem
        database = problem.database
        formulation.compute_samples(
            problem, input_data, compute_jacobian=compute_jacobian
        )
        for output_name in database.get_function_names(skip_grad=False):
            output_data[output_name] = database.get_function_history(output_name)
