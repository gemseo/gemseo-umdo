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
"""A function to compute a statistic from `PCE`.

See also [PCE][gemseo_umdo.formulations.pce.PCE].
"""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

from gemseo.mlearning.regression.algos.pce import PCERegressor
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays as array_to_dict

from gemseo_umdo.formulations._functions.statistic_function_for_surrogate import (
    StatisticFunctionForSurrogate,
)
from gemseo_umdo.formulations._statistics.pce.base_pce_estimator import BasePCEEstimator

if TYPE_CHECKING:
    from gemseo.typing import NumberArray
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.pce import PCE


PCET = TypeVar("PCET", bound="PCE")


class StatisticFunctionForPCE(StatisticFunctionForSurrogate[PCET]):
    """A function to compute a statistic from `PCE`."""

    def _compute_statistic_estimation(
        self, output_data: dict[str, PCERegressor]
    ) -> RealArray:
        name = self._function_name
        estimator = self._statistic_estimator
        return estimator.estimate_statistic(*[
            output_data[key][name] for key in estimator.ARG_NAMES
        ])

    def _compute_output_data(
        self,
        input_data: RealArray,
        output_data: dict[str, dict[str, NumberArray]],
        compute_jacobian: bool = False,
    ) -> None:
        pce_formulation = self._umdo_formulation
        problem = pce_formulation.mdo_formulation.optimization_problem
        samples = pce_formulation.compute_samples(problem)
        pce_regressor = PCERegressor(
            samples, settings_model=pce_formulation._settings.regressor_settings
        )
        pce_regressor.learn()

        # Store the output data.
        sizes = pce_regressor.sizes
        output_names = pce_regressor.output_names
        output_data[BasePCEEstimator.MEAN_ARG_NAME] = array_to_dict(
            pce_regressor.mean, sizes, output_names
        )
        output_data[BasePCEEstimator.STD_ARG_NAME] = array_to_dict(
            pce_regressor.standard_deviation, sizes, output_names
        )
        output_data[BasePCEEstimator.VAR_ARG_NAME] = array_to_dict(
            pce_regressor.variance, sizes, output_names
        )
        self._log_regressor_quality(pce_regressor)
