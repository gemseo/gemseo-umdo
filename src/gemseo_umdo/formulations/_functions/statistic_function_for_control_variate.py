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
"""A function to compute a statistic from `ControlVariate`.

See also [ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate].
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import TypeVar

from gemseo.mlearning.regression.algos.factory import RegressorFactory
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import diag
from numpy import diagonal
from numpy import newaxis
from numpy.linalg import multi_dot

from gemseo_umdo.formulations._functions.base_statistic_function import (
    BaseStatisticFunction,
)

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.control_variate import ControlVariate

ControlVariateT = TypeVar("ControlVariateT", bound="ControlVariate")


class StatisticFunctionForControlVariate(BaseStatisticFunction[ControlVariateT]):
    """A function to compute a statistic from `ControlVariate`."""

    __DOE_PRED_TEMPLATE: Final[str] = "pred_{}"
    """The template for the prediction history name."""

    __LARGE_DOE_PRED_TEMPLATE: Final[str] = "large_pred_{}"
    """The template for the large prediction history name."""

    __MEAN_TEMPLATE: Final[str] = "mean_{}"
    """The template for the name of the mean of a regressor output."""

    __VARIANCE_TEMPLATE: Final[str] = "variance_{}"
    """The template for the name of the variance of a regressor output."""

    __mc_problem: OptimizationProblem
    """The evaluation problem defined over the uncertain space for sampling."""

    __input_mean: RealArray
    """The mean of the uncertain input vector."""

    __input_variance: RealArray
    """The variance of the uncertain input vector."""

    __problem: OptimizationProblem
    """The evaluation problem defined over the uncertain space for Taylor."""

    def __init__(
        self,
        umdo_formulation: ControlVariateT,
        output_name: str,
        function_type: MDOFunction.FunctionType,
        statistic_operator_name: str,
        **statistic_options: Any,
    ) -> None:
        super().__init__(
            umdo_formulation,
            output_name,
            function_type,
            statistic_operator_name,
            **statistic_options,
        )
        self.__mc_problem = self._umdo_formulation.mdo_formulation.optimization_problem
        formulation = self._umdo_formulation
        distribution = formulation.uncertain_space.distribution
        self.__input_mean = distribution.mean
        self.__input_variance = distribution.standard_deviation**2
        self.__problem = formulation.auxiliary_mdo_formulation.optimization_problem

    @property
    def _statistic_estimator_parameters(self) -> tuple[ParameterSpace]:
        return (self._umdo_formulation.uncertain_space,)

    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, Any]:
        umdo_formulation = self._umdo_formulation
        settings = umdo_formulation._settings
        compute_samples = umdo_formulation.compute_samples
        problem = self.__mc_problem
        if settings.regressor_settings is not None:
            compute_samples(problem, settings.regressor_doe_algo_settings)
            samples = problem.to_dataset(opt_naming=False, export_gradients=True)
            problem.reset(preprocessing=False)
            regressor_settings = settings.regressor_settings
            regressor = RegressorFactory().create(
                regressor_settings.__class__.__name__.rsplit("_Settings", 1)[0],
                samples,
                settings_model=regressor_settings,
            )
            regressor.learn()

        compute_samples(problem)
        get_history = problem.database.get_function_history
        mean_template = self.__MEAN_TEMPLATE
        variance_template = self.__VARIANCE_TEMPLATE
        doe_pred_template = self.__DOE_PRED_TEMPLATE
        large_doe_pred_template = self.__LARGE_DOE_PRED_TEMPLATE
        data = {}
        for function in self.__problem.functions:
            output_name = function.name
            data[output_name] = get_history(output_name)

        uncertain_space = umdo_formulation.uncertain_space
        doe = self._umdo_formulation.doe_algo.samples
        large_doe = uncertain_space.compute_samples(10000)
        if settings.regressor_settings is None:
            input_variance = self.__input_variance
            mean_input_value = self.__input_mean
            for function in self.__problem.functions:
                output_name = function.name
                mean = atleast_1d(function.evaluate(mean_input_value))
                jac = atleast_2d(function.jac(mean_input_value))
                variance = diagonal(multi_dot([jac, diag(input_variance), jac.T]))
                data[mean_template.format(output_name)] = mean
                data[variance_template.format(output_name)] = variance
                data[doe_pred_template.format(output_name)] = (
                    mean + (doe - mean_input_value) @ jac.T
                )
                data[large_doe_pred_template.format(output_name)] = (
                    mean + (large_doe - mean_input_value) @ jac.T
                )
        else:
            doe = split_array_to_dict_of_arrays(
                doe,
                uncertain_space.variable_sizes,
                uncertain_space.variable_names,
            )
            large_doe = split_array_to_dict_of_arrays(
                large_doe,
                uncertain_space.variable_sizes,
                uncertain_space.variable_names,
            )
            doe_pred = regressor.predict(doe)
            large_doe_pred = regressor.predict(large_doe)
            for function in self.__problem.functions:
                output_name = function.name
                doe_pred_ = doe_pred[output_name]
                large_doe_pred_ = large_doe_pred[output_name]
                data[doe_pred_template.format(output_name)] = doe_pred_
                data[large_doe_pred_template.format(output_name)] = large_doe_pred_
                data[mean_template.format(output_name)] = doe_pred_.mean(0)
                data[variance_template.format(output_name)] = large_doe_pred_.var(0)

        return data

    def _compute_statistic_estimation(self, data: dict[str, RealArray]) -> RealArray:
        output_name = self._output_name
        samples = data[output_name]
        if samples.ndim == 1:
            samples = samples[:, newaxis]

        return self._statistic_estimator.estimate_statistic(
            samples,
            data[self.__MEAN_TEMPLATE.format(output_name)],
            data[self.__VARIANCE_TEMPLATE.format(output_name)],
            data[self.__DOE_PRED_TEMPLATE.format(output_name)],
            data[self.__LARGE_DOE_PRED_TEMPLATE.format(output_name)],
        )
