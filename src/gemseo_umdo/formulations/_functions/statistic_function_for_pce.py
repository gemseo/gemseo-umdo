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
        function_name = self._function_name
        estimator = self._statistic_estimator
        return estimator.estimate_statistic(*[
            output_data[statistic_name][function_name]
            for statistic_name in estimator.ARG_NAMES
        ])

    def _compute_statistic_jacobian_estimation(
        self, output_data: dict[str, RealArray]
    ) -> RealArray:
        function_name = self._function_name
        estimator = self._statistic_estimator
        return estimator.compute_jacobian(*[
            output_data[self.__get_statistic_jac_name(statistic_name)][function_name]
            for statistic_name in estimator.ARG_NAMES
        ])

    def _compute_output_data(
        self,
        input_data: RealArray,
        output_data: dict[str, dict[str, NumberArray]],
        compute_jacobian: bool = False,
    ) -> None:
        pce_formulation = self._umdo_formulation
        problem = pce_formulation.mdo_formulation.optimization_problem
        samples = pce_formulation.compute_samples(
            problem, compute_jacobian=compute_jacobian
        )
        pce_regressor = PCERegressor(
            samples, settings_model=pce_formulation._settings.regressor_settings
        )
        pce_regressor.learn()

        # Store the output data.
        sizes = pce_regressor.sizes
        output_names = pce_regressor.output_names
        mean_arg_name = BasePCEEstimator.MEAN_ARG_NAME
        std_arg_name = BasePCEEstimator.STD_ARG_NAME
        var_arg_name = BasePCEEstimator.VAR_ARG_NAME
        output_data[mean_arg_name] = array_to_dict(
            pce_regressor.mean, sizes, output_names
        )
        output_data[std_arg_name] = array_to_dict(
            pce_regressor.standard_deviation, sizes, output_names
        )
        output_data[var_arg_name] = array_to_dict(
            pce_regressor.variance, sizes, output_names
        )
        if compute_jacobian:
            output_data[self.__get_statistic_jac_name(mean_arg_name)] = array_to_dict(
                pce_regressor.mean_jacobian_wrt_special_variables, sizes, output_names
            )
            output_data[self.__get_statistic_jac_name(std_arg_name)] = array_to_dict(
                pce_regressor.standard_deviation_jacobian_wrt_special_variables,
                sizes,
                output_names,
            )
            output_data[self.__get_statistic_jac_name(var_arg_name)] = array_to_dict(
                pce_regressor.variance_jacobian_wrt_special_variables,
                sizes,
                output_names,
            )
        self._log_regressor_quality(pce_regressor)

    @staticmethod
    def __get_statistic_jac_name(statistic_name: str) -> str:
        """Return the name of the Jacobian of the statistic.

        Args:
            statistic_name: The name of the statistic.

        Returns:
            The name of the Jacobian of the statistic.
        """
        return f"{statistic_name}_jac"
