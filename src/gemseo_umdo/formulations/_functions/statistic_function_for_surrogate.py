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
"""A function to compute a statistic from `Surrogate`.

See also [Surrogate][gemseo_umdo.formulations.surrogate.Surrogate].
"""  # noqa: E501

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

from gemseo.mlearning.regression.algos.factory import RegressorFactory

from gemseo_umdo.formulations._functions.base_statistic_function import (
    BaseStatisticFunction,
)
from gemseo_umdo.formulations._functions.base_statistic_function import UMDOFormulationT

if TYPE_CHECKING:
    from gemseo.algos.database import DatabaseKeyType
    from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.surrogate import Surrogate

LOGGER = logging.getLogger(__name__)

SurrogateT = TypeVar("SurrogateT", bound="Surrogate")


class StatisticFunctionForSurrogate(BaseStatisticFunction[SurrogateT]):
    """A function to compute a statistic from `Surrogate`."""

    __output_names_to_qualities: dict[str, RealArray]
    """The surrogate model quality for the different output variables."""

    def __init__(
        self,
        umdo_formulation: UMDOFormulationT,
        output_name: str,
        function_type: BaseStatisticFunction.FunctionType,
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
        self.__output_names_to_qualities = {}

    def _compute_statistic_estimation(self, data: dict[str, RealArray]) -> RealArray:
        return self._statistic_estimator.estimate_statistic(data[self._output_name])

    def _compute_data_for_statistic_estimation(
        self, input_data: RealArray, estimate_jacobian: bool
    ) -> dict[str, Any]:
        umdo_formulation = self._umdo_formulation
        problem = umdo_formulation.mdo_formulation.optimization_problem
        samples = umdo_formulation.compute_samples(problem)
        regressor_settings = umdo_formulation._settings.regressor_settings
        regressor = RegressorFactory().create(
            regressor_settings.__class__.__name__.rsplit("_Settings", 1)[0],
            samples,
            settings_model=regressor_settings,
        )
        regressor.learn()
        output_samples = regressor.predict(umdo_formulation.input_samples)
        self._log_regressor_quality(regressor, input_data)
        return output_samples

    def _log_regressor_quality(
        self, regressor: BaseRegressor, input_data: RealArray
    ) -> None:
        """Log the quality of the regressor.

        Args:
            regressor: The regressor.
            input_data: The input point.
        """
        surrogate_formulation = self._umdo_formulation
        quality = surrogate_formulation.quality(regressor)
        names_to_learning_measures = quality.compute_learning_measure(as_dict=True)
        LOGGER.info("        %s", quality.__class__.__name__)
        quality_cv_options = surrogate_formulation.quality_cv_options
        is_surrogate_quality_bad = surrogate_formulation.is_surrogate_quality_bad
        quality_operators = surrogate_formulation.quality_operators
        threshold = surrogate_formulation.threshold
        cv_threshold = surrogate_formulation.cv_threshold
        names_to_qualities = self.__output_names_to_qualities
        if quality_cv_options:
            names_to_test_measures = quality.compute_cross_validation_measure(
                as_dict=True, **quality_cv_options
            )
            for output_name, learning_measure in names_to_learning_measures.items():
                test_measure = names_to_test_measures[output_name]
                thresh = threshold[output_name]
                cv_thresh = cv_threshold[output_name]
                names_to_qualities[f"{output_name}_learning_quality"] = learning_measure
                names_to_qualities[f"{output_name}_test_quality"] = test_measure
                for index, l_measure_i in enumerate(learning_measure):
                    t_measure_i = test_measure[index]
                    train_is_bad = is_surrogate_quality_bad(l_measure_i, thresh[index])
                    cv_is_bad = is_surrogate_quality_bad(t_measure_i, cv_thresh[index])
                    if train_is_bad or cv_is_bad:
                        level = logging.WARNING
                    else:
                        level = logging.INFO
                    LOGGER.log(
                        level,
                        "            %s[%s]: %s%s%s (learning) - %s%s%s (test)",
                        output_name,
                        index,
                        l_measure_i,
                        quality_operators[int(train_is_bad)],
                        thresh[index],
                        t_measure_i,
                        quality_operators[int(cv_is_bad)],
                        cv_thresh[index],
                    )
        else:
            for output_name, learning_measure in names_to_learning_measures.items():
                thresh = threshold[output_name]
                names_to_qualities[f"{output_name}_learning_quality"] = learning_measure
                for index, l_measure_i in enumerate(learning_measure):
                    train_is_bad = is_surrogate_quality_bad(l_measure_i, thresh[index])
                    LOGGER.log(
                        logging.WARNING if train_is_bad else logging.INFO,
                        "            %s[%s]: %s%s%s (learning)",
                        output_name,
                        index,
                        l_measure_i,
                        quality_operators[int(train_is_bad)],
                        thresh[index],
                    )

        surrogate_formulation.optimization_problem.database.add_new_iter_listener(
            self._store_surrogate_model_quality
        )

    def _store_surrogate_model_quality(self, input_data: DatabaseKeyType) -> None:
        """Store the quality of the surrogate model in the database.

        Args:
            input_data: The input point.
        """
        self._umdo_formulation.optimization_problem.database.store(
            input_data, self.__output_names_to_qualities
        )
