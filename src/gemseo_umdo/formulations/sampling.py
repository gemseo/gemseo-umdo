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
r"""Sampling-based U-MDO formulation.

[Sampling][gemseo_umdo.formulations.sampling.Sampling] is an
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
estimating the statistics with (quasi) Monte Carlo techniques.

E.g.

$$\mathbb{E}[f(x,U)] \approx \frac{1}{N}\sum_{i=1}^N f\left(x,U^{(i)}\right)$$

or

$$\mathbb{V}[f(x,U)] \approx
\frac{1}{N}\sum_{i=1}^N \left(f\left(x,U^{(i)}\right)-
\frac{1}{N}\sum_{j=1}^N f\left(x,U^{(j)}\right)\right)^2$$

where $U$ is normally distributed
with mean $\mu$ and variance $\sigma^2$
and $U^{(1)},\ldots,U^{(N)}$ are $N$ realizations of $U$
obtained with an optimized Latin hypercube sampling technique.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo import to_pickle
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.logging_tools import LoggingContext

from gemseo_umdo.formulations._functions.statistic_function_for_iterative_sampling import (  # noqa: E501
    StatisticFunctionForIterativeSampling,
)
from gemseo_umdo.formulations._functions.statistic_function_for_standard_sampling import (  # noqa: E501
    StatisticFunctionForStandardSampling,
)
from gemseo_umdo.formulations._statistics.iterative_sampling.factory import (  # noqa: E501
    SamplingEstimatorFactory as IterativeSamplingEstimatorFactory,
)
from gemseo_umdo.formulations._statistics.sampling.factory import (
    SamplingEstimatorFactory,
)
from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.base_doe_library import BaseDOELibrary
    from gemseo.algos.doe.base_doe_library import CallbackType
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


class Sampling(BaseUMDOFormulation):
    """Sampling-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the
        [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html).
        for more information about the available DOE algorithm names and options.
    """

    Settings: ClassVar[type[Sampling_Settings]] = Sampling_Settings

    __doe_algo: BaseDOELibrary
    """The DOE library to execute the DOE algorithm."""

    __samples_directory_path: str | Path
    """The path to the directory where the samples are saved."""

    callbacks: list[CallbackType]
    """The callback functions for the DOE algorithm when computing the output data."""

    jacobian_callbacks: list[CallbackType]
    """The callback functions for the DOE algorithm when computing the Jacobian."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        settings_model: Sampling_Settings,
        objective_statistic_parameters: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        mdo_formulation_settings: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> None:
        self.callbacks = []
        self.jacobian_callbacks = []
        self.input_data_to_output_samples = {}
        if settings_model.samples_directory_path:
            self.__samples_directory_path = Path(settings_model.samples_directory_path)
            self.__samples_directory_path.mkdir()
            settings_model.estimate_statistics_iteratively = False
        else:
            self.__samples_directory_path = ""

        estimate_statistics_iteratively = settings_model.estimate_statistics_iteratively
        if estimate_statistics_iteratively:
            self._statistic_factory = IterativeSamplingEstimatorFactory()
            self._statistic_function_class = StatisticFunctionForIterativeSampling
        else:
            self._statistic_factory = SamplingEstimatorFactory()
            self._statistic_function_class = StatisticFunctionForStandardSampling

        doe_algo_settings = settings_model.doe_algo_settings
        doe_algo_settings.use_database = not estimate_statistics_iteratively
        algo_name = doe_algo_settings._TARGET_CLASS_NAME
        self.__doe_algo = DOELibraryFactory().create(algo_name)
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            settings_model,
            objective_statistic_parameters=objective_statistic_parameters,
            mdo_formulation_settings=mdo_formulation_settings,
        )
        mdo_formulation = self._mdo_formulation.__class__.__name__
        formulation = self.__class__.__name__
        self.name = f"{formulation}[{mdo_formulation}; {algo_name}"
        if "n_samples" in doe_algo_settings.model_fields:
            self.name += f"({doe_algo_settings.n_samples})]"

    def compute_samples(
        self,
        problem: OptimizationProblem,
        input_data: RealArray,
        compute_jacobian: bool = False,
    ) -> None:
        """Evaluate the functions of a problem with a DOE algorithm.

        Args:
            problem: The sampling problem.
            input_data: The input point at which to estimate the statistic.
            compute_jacobian: Whether to compute the Jacobian of the objective.
        """
        doe_algo_settings = self._settings.doe_algo_settings
        original_callbacks = doe_algo_settings.callbacks
        new_callbacks = self.jacobian_callbacks if compute_jacobian else self.callbacks
        doe_algo_settings.callbacks = list(original_callbacks) + new_callbacks
        doe_algo_settings.eval_jac = compute_jacobian
        with LoggingContext(logging.getLogger("gemseo")):
            self.__doe_algo.execute(
                problem, eval_obs_jac=compute_jacobian, settings_model=doe_algo_settings
            )

        if self.__samples_directory_path:
            main_problem = self.optimization_problem
            iteration = main_problem.evaluation_counter.current + 1
            dataset = problem.to_dataset(f"Iteration {iteration}", opt_naming=False)
            dataset.misc.update(
                main_problem.design_space.convert_array_to_dict(input_data)
            )
            to_pickle(dataset, self.__samples_directory_path / f"{iteration}.pkl")

        doe_algo_settings.callbacks = original_callbacks
