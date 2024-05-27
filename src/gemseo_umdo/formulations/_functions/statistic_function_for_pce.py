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

import logging
from typing import TYPE_CHECKING
from typing import TypeVar

from gemseo.mlearning.regression.algos.pce import PCERegressor
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays as array_to_dict

from gemseo_umdo.formulations._functions.base_statistic_function import (
    BaseStatisticFunction,
)
from gemseo_umdo.formulations._statistics.pce.base_pce_estimator import BasePCEEstimator

if TYPE_CHECKING:
    from gemseo.typing import NumberArray
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.pce import PCE

LOGGER = logging.getLogger(__name__)

PCET = TypeVar("PCET", bound="PCE")


class StatisticFunctionForPCE(BaseStatisticFunction[PCET]):
    """A function to compute a statistic from `PCE`."""

    def _compute_statistic_estimation(
        self, output_data: dict[str, PCERegressor]
    ) -> RealArray:
        name = self._function_name
        estimator = self._estimate_statistic
        return estimator(*[output_data[key][name] for key in estimator.ARG_NAMES])

    def _compute_output_data(
        self,
        input_data: RealArray,
        output_data: dict[str, PCERegressor | dict[str, NumberArray]],
    ) -> None:
        pce_formulation = self._formulation
        problem = pce_formulation.mdo_formulation.optimization_problem
        samples = pce_formulation.compute_samples(problem)
        pce_regressor = PCERegressor(
            samples,
            pce_formulation.mdo_formulation.design_space,
            **pce_formulation.pce_options,
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
        problem.reset(preprocessing=False)
        self.__log_pce_quality(pce_regressor)

    def __log_pce_quality(self, pce: PCERegressor) -> None:
        """Log the quality of the PCE regressor.

        Args:
            pce: The PCE regressor.
        """
        pce_formulation = self._formulation
        quality = pce_formulation.quality(pce)
        train = quality.compute_learning_measure(as_dict=True)
        LOGGER.info("        %s", quality.__class__.__name__)
        quality_cv_options = pce_formulation.quality_cv_options
        is_pce_quality_bad = pce_formulation.is_pce_quality_bad
        quality_operators = pce_formulation.quality_operators
        threshold = pce_formulation.threshold
        cv_threshold = pce_formulation.cv_threshold
        if quality_cv_options:
            test = quality.compute_cross_validation_measure(
                as_dict=True, **quality_cv_options
            )
            for output_name, train_values in train.items():
                test_values = test[output_name]
                thresh = threshold[output_name]
                cv_thresh = cv_threshold[output_name]
                for index, train_value in enumerate(train_values):
                    test_value = test_values[index]
                    train_is_bad = is_pce_quality_bad(train_value, thresh[index])
                    cv_is_bad = is_pce_quality_bad(test_value, cv_thresh[index])
                    if train_is_bad or cv_is_bad:
                        level = logging.WARNING
                    else:
                        level = logging.INFO
                    LOGGER.log(
                        level,
                        "            %s[%s]: %s%s%s (train) - %s%s%s (test)",
                        output_name,
                        index,
                        train_value,
                        quality_operators[train_is_bad],
                        thresh[index],
                        test_value,
                        quality_operators[cv_is_bad],
                        cv_thresh[index],
                    )
        else:
            for output_name, train_values in train.items():
                thresh = threshold[output_name]
                for index, train_value in enumerate(train_values):
                    train_is_bad = is_pce_quality_bad(train_value, thresh[index])
                    LOGGER.log(
                        logging.WARNING if train_is_bad else logging.INFO,
                        "            %s[%s]: %s%s%s (train)",
                        output_name,
                        index,
                        train_value,
                        quality_operators[int(train_is_bad)],
                        thresh[index],
                    )
