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
from typing import TypeVar

from gemseo.mlearning.regression.algos.factory import RegressorFactory

from gemseo_umdo.formulations._functions.base_statistic_function import (
    BaseStatisticFunction,
)

if TYPE_CHECKING:
    from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.surrogate import Surrogate

LOGGER = logging.getLogger(__name__)

SurrogateT = TypeVar("SurrogateT", bound="Surrogate")


class StatisticFunctionForSurrogate(BaseStatisticFunction[SurrogateT]):
    """A function to compute a statistic from `Surrogate`."""

    def _compute_statistic_estimation(
        self, output_data: dict[str, RealArray]
    ) -> RealArray:
        return self._statistic_estimator.estimate_statistic(
            output_data[self._function_name]
        )

    def _compute_output_data(
        self,
        input_data: RealArray,
        output_data: dict[str, RealArray],
        compute_jacobian: bool = False,
    ) -> None:
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
        output_data.update(regressor.predict(umdo_formulation.input_samples))
        self._log_regressor_quality(regressor)

    def _log_regressor_quality(self, regressor: BaseRegressor) -> None:
        """Log the quality of the regressor.

        Args:
            regressor: The regressor.
        """
        surrogate_formulation = self._umdo_formulation
        quality = surrogate_formulation.quality(regressor)
        train = quality.compute_learning_measure(as_dict=True)
        LOGGER.info("        %s", quality.__class__.__name__)
        quality_cv_options = surrogate_formulation.quality_cv_options
        is_surrogate_quality_bad = surrogate_formulation.is_surrogate_quality_bad
        quality_operators = surrogate_formulation.quality_operators
        threshold = surrogate_formulation.threshold
        cv_threshold = surrogate_formulation.cv_threshold
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
                    train_is_bad = is_surrogate_quality_bad(train_value, thresh[index])
                    cv_is_bad = is_surrogate_quality_bad(test_value, cv_thresh[index])
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
                        quality_operators[int(train_is_bad)],
                        thresh[index],
                        test_value,
                        quality_operators[int(cv_is_bad)],
                        cv_thresh[index],
                    )
        else:
            for output_name, train_values in train.items():
                thresh = threshold[output_name]
                for index, train_value in enumerate(train_values):
                    train_is_bad = is_surrogate_quality_bad(train_value, thresh[index])
                    LOGGER.log(
                        logging.WARNING if train_is_bad else logging.INFO,
                        "            %s[%s]: %s%s%s (train)",
                        output_name,
                        index,
                        train_value,
                        quality_operators[int(train_is_bad)],
                        thresh[index],
                    )
