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
from typing import Any
from typing import TypeVar

from gemseo.mlearning.regression.algos.factory import RegressorFactory
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays as array_to_dict
from numpy import array
from numpy import hstack
from scipy.linalg import solve

from gemseo_umdo.formulations._functions.statistic_function_for_surrogate import (
    StatisticFunctionForSurrogate,
)
from gemseo_umdo.formulations._statistics.pce.base_pce_estimator import BasePCEEstimator

if TYPE_CHECKING:
    from gemseo.mlearning.regression.algos.base_fce import BaseFCERegressor
    from gemseo.mlearning.regression.algos.pce import PCERegressor
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.pce import PCE


PCET = TypeVar("PCET", bound="PCE")


class StatisticFunctionForPCE(StatisticFunctionForSurrogate[PCET]):
    """A function to compute a statistic from `PCE`."""

    def _compute_statistic_estimation(self, data: dict[str, PCERegressor]) -> RealArray:
        function_name = self._output_name
        estimator = self._statistic_estimator
        return estimator.estimate_statistic(*[
            data[statistic_name][function_name]
            for statistic_name in estimator.ARG_NAMES
        ])

    def _compute_statistic_jacobian_estimation(
        self, data: dict[str, RealArray]
    ) -> RealArray:
        function_name = self._output_name
        estimator = self._statistic_estimator
        return estimator.compute_jacobian(*[
            data[self.__get_statistic_jac_name(statistic_name)][function_name]
            for statistic_name in estimator.ARG_NAMES
        ])

    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, Any]:
        pce_formulation = self._umdo_formulation
        settings = pce_formulation._settings
        regressor_settings = settings.regressor_settings
        if regressor_settings.learn_jacobian_data:
            # FCE(u) trained from samples of df(x,u)du (i.e. gradient-enhanced FCE)
            problem = pce_formulation.auxiliary_mdo_formulation.optimization_problem
            compute_jacobian = True
        elif estimate_jacobian and not settings.approximate_statistics_jacobians:
            # FCE(u) trained from samples of df(x,u)dx
            problem = pce_formulation.mdo_formulation.optimization_problem
            compute_jacobian = True
        else:
            # Standard FCE(u) trained without samples of derivatives
            problem = pce_formulation.mdo_formulation.optimization_problem
            compute_jacobian = False

        samples = pce_formulation.compute_samples(problem, compute_jacobian)

        if estimate_jacobian and not settings.approximate_statistics_jacobians:
            regressor_settings.use_special_jacobian_data = True
        else:
            regressor_settings.use_special_jacobian_data = False

        fce = RegressorFactory().create(
            regressor_settings._TARGET_CLASS_NAME,
            samples,
            settings_model=regressor_settings,
        )
        fce.learn()

        # Store the output data.
        sizes = fce.sizes
        output_names = fce.output_names
        mean_arg_name = BasePCEEstimator.MEAN_ARG_NAME
        std_arg_name = BasePCEEstimator.STD_ARG_NAME
        var_arg_name = BasePCEEstimator.VAR_ARG_NAME
        data = {
            mean_arg_name: array_to_dict(fce.mean, sizes, output_names),
            std_arg_name: array_to_dict(fce.standard_deviation, sizes, output_names),
            var_arg_name: array_to_dict(fce.variance, sizes, output_names),
        }
        if estimate_jacobian:
            if settings.approximate_statistics_jacobians:
                jac_mean, jac_std, jac_var = self.__approximate_jacobians(fce)
            else:
                jac_mean = fce.mean_jacobian_wrt_special_variables
                jac_std = fce.standard_deviation_jacobian_wrt_special_variables
                jac_var = fce.variance_jacobian_wrt_special_variables

            jac_mean = {
                k: v.T
                for k, v in array_to_dict(jac_mean.T, sizes, output_names).items()
            }
            jac_std = {
                k: v.T for k, v in array_to_dict(jac_std.T, sizes, output_names).items()
            }
            jac_var = {
                k: v.T for k, v in array_to_dict(jac_var.T, sizes, output_names).items()
            }
            data[self.__get_statistic_jac_name(mean_arg_name)] = jac_mean
            data[self.__get_statistic_jac_name(std_arg_name)] = jac_std
            data[self.__get_statistic_jac_name(var_arg_name)] = jac_var

        self._log_regressor_quality(fce, input_data)
        return data

    @staticmethod
    def __get_statistic_jac_name(statistic_name: str) -> str:
        """Return the name of the Jacobian of the statistic.

        Args:
            statistic_name: The name of the statistic.

        Returns:
            The name of the Jacobian of the statistic.
        """
        return f"{statistic_name}_jac"

    def __approximate_jacobians(
        self, fce_regressor: BaseFCERegressor
    ) -> tuple[RealArray, RealArray, RealArray]:
        """Approximate the Jacobians of mean, variance and standard deviation.

        Args:
            fce_regressor: A functional chaos expansion regressor.

        Returns:
            The Jacobian of the mean,
            the Jacobian of the standard deviation
            and the Jacobian of the variance.
        """
        differentiation_step = self._umdo_formulation._settings.differentiation_step

        mean_down = []
        var_down = []
        mean_up = []
        var_up = []

        data = fce_regressor.learning_set
        indices = fce_regressor.learning_samples_indices
        input_sample = data.input_dataset.get_view(indices=indices).to_numpy()
        output_sample = data.output_dataset.get_view(indices=indices).to_numpy()
        basis_functions = fce_regressor._basis_functions
        transformation = fce_regressor._isoprobabilistic_transformation

        # Original training set: (u_i(1), ..., u_i(d), y_i)_{i=1...N}
        i = 0
        for input_name in fce_regressor.input_names:
            for _ in range(fce_regressor.sizes[input_name]):
                input_sample_i = input_sample.copy()
                for step, mean_, var_ in (
                    # (u_i(1), ..., u_i(j-1), u(j)+ε, u_i(j+1), ..., u_i(d), y_i)_i
                    (differentiation_step, mean_down, var_down),
                    # (u_i(1), ..., u_i(j-1), u(j)-2ε, u_i(j+1), ..., u_i(d), y_i)_i
                    (-2 * differentiation_step, mean_up, var_up),
                ):
                    input_sample_i[:, i] += step
                    phi = hstack([
                        array(basis_function(transformation(input_sample_i)))
                        for basis_function in basis_functions
                    ])
                    coefficients = (
                        solve(
                            phi.T @ phi,
                            phi.T,
                            overwrite_a=True,
                            overwrite_b=True,
                            assume_a="sym",
                        )
                        @ output_sample
                    )
                    mean_.append(coefficients[0, :])
                    var_.append((coefficients[1:, :] ** 2).sum(axis=0))

                i += 1

        mean_down = array(mean_down).T
        var_down = array(var_down).T
        mean_up = array(mean_up).T
        var_up = array(var_up).T

        # statistic_jacobian = (stat(x+ε)-stat(x-ε))/(2ε)
        two_steps = 2 * differentiation_step
        mean_jacobian = (mean_up - mean_down) / two_steps
        variance_jacobian = (var_up - var_down) / two_steps
        standard_deviation_jacobian = (var_up**0.5 - var_down**0.5) / two_steps
        return mean_jacobian, standard_deviation_jacobian, variance_jacobian
