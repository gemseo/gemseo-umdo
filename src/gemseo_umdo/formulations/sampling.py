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

[Sampling][gemseo_umdo.formulations.sampling.Sampling] is a
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
from gemseo.doe.factory import DOELibraryFactory
from gemseo.util.logging import LoggingContext
from gemseo.util.pydantic import create_model

from gemseo_umdo.formulations._functions.statistic_function_for_sampling import (
    StatisticFunctionForSampling,
)
from gemseo_umdo.formulations._statistics.iterative_sampling.factory import (  # noqa: E501
    SamplingEstimatorFactory as IterativeSamplingEstimatorFactory,
)
from gemseo_umdo.formulations._statistics.sampling.factory import (
    SamplingEstimatorFactory as OneShotSamplingEstimatorFactory,
)
from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.problem.evaluation import EvaluationProblem
    from gemseo.discipline import Discipline
    from gemseo.doe.core.base_doe_library import BaseDOELibrary
    from gemseo.doe.core.base_doe_library import CallbackType
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import RandomSpace
    from gemseo.util.typing import RealArray

    from gemseo_umdo.formulations._functions.iterative_estimation import (
        IterativeEstimation,
    )


class Sampling(BaseUMDOFormulation):
    """Sampling-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the documentation of the [DOE algorithms][gemseo.doe]
        for more information about the available DOE algorithms and their settings.
    """

    settings_class: ClassVar[type[Sampling_Settings]] = Sampling_Settings

    _statistic_function_class: type[StatisticFunctionForSampling] = (
        StatisticFunctionForSampling
    )

    __doe_algo: BaseDOELibrary
    """The DOE library to execute the DOE algorithm."""

    __samples_directory_path: str | Path
    """The path to the directory where the samples are saved."""

    callbacks: list[CallbackType]
    """The callback functions for the DOE algorithm when computing the output data."""

    jacobian_callbacks: list[CallbackType]
    """The callback functions for the DOE algorithm when computing the Jacobian."""

    _estimate_statistics_iteratively: bool
    """Whether the statistics are estimated iteratively."""

    _statistic_functions: list[StatisticFunctionForSampling]
    """The functions computing the statistics iteratively during the sampling.

    A function takes part in the sampling
    once created by `create_objective` or `create_constraint`
    or added by `add_constraint` or `add_observable`,
    and never stops taking part in it,
    e.g. an objective replaced by a new one created by `create_objective`.
    """

    def __init__(  # noqa: D107
        self,
        problem: OptimizationProblem,
        disciplines: Sequence[Discipline],
        settings: Sampling_Settings | None = None,
        *,
        uncertain_space: RandomSpace,
        mdo_formulation_settings: BaseFormulationSettings | None = None,
    ) -> None:
        settings = create_model(self.settings_class, settings_model=settings)
        self.callbacks = []
        self.jacobian_callbacks = []
        self._statistic_functions = []
        self.input_data_to_output_samples = {}
        if settings.samples_directory_path:
            self.__samples_directory_path = Path(settings.samples_directory_path)
            self.__samples_directory_path.mkdir()
            settings.estimate_statistics_iteratively = False
        else:
            self.__samples_directory_path = ""

        self._estimate_statistics_iteratively = settings.estimate_statistics_iteratively
        self._statistic_factory = (
            IterativeSamplingEstimatorFactory()
            if self._estimate_statistics_iteratively
            else OneShotSamplingEstimatorFactory()
        )
        doe_algo_settings = settings.doe_algo_settings
        algo_name = doe_algo_settings.target_class_name
        self.__doe_algo = DOELibraryFactory().create(algo_name)
        super().__init__(
            problem,
            disciplines,
            settings=settings,
            uncertain_space=uncertain_space,
            mdo_formulation_settings=mdo_formulation_settings,
        )
        mdo_formulation = self._mdo_formulation.__class__.__name__
        formulation = self.__class__.__name__
        self.name = f"{formulation}[{mdo_formulation}; {algo_name}"
        if "n_samples" in type(doe_algo_settings).model_fields:
            self.name += f"({doe_algo_settings.n_samples})]"

    @property
    def doe_algo(self) -> BaseDOELibrary:
        """The DOE library configured with an algorithm."""
        return self.__doe_algo

    def _check_statistic_name(self, statistic_name: str) -> None:
        """
        Raises:
            ValueError: When the statistic cannot be estimated iteratively
                whereas the statistics are estimated iteratively.
        """  # noqa: D205 D212 D415
        if (
            self._estimate_statistics_iteratively
            and not self._statistic_factory.is_available(statistic_name)
            and OneShotSamplingEstimatorFactory().is_available(statistic_name)
        ):
            msg = (
                f"The statistic '{statistic_name}' cannot be estimated "
                'iteratively; use "estimate_statistics_iteratively=False".'
            )
            raise ValueError(msg)

        super()._check_statistic_name(statistic_name)

    def _register_statistic_function(  # noqa: D102
        self, function: StatisticFunctionForSampling
    ) -> None:
        if not self._estimate_statistics_iteratively:
            return

        # The cached statistics have been estimated without this function.
        self.input_data_to_output_data.clear()
        self._statistic_functions.append(function)

    def _post_add_mdo_observable(self) -> None:
        # The cached data have been computed without this MDO observable.
        self.input_data_to_output_data.clear()

    def get_iterative_estimations(self, jacobian: bool) -> list[IterativeEstimation]:
        """Return the iterative estimations of the statistics or of their Jacobians.

        To be used only when the statistics are estimated iteratively.

        Args:
            jacobian: Whether to return the estimations of the Jacobians.

        Returns:
            The iterative estimations of the statistics or of their Jacobians.
        """
        return [
            function.get_iterative_estimation(jacobian)
            for function in self._statistic_functions
        ]

    def compute_samples(
        self,
        problem: EvaluationProblem,
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
        estimate_statistics_iteratively = self._estimate_statistics_iteratively
        if estimate_statistics_iteratively:
            estimations = self.get_iterative_estimations(compute_jacobian)
            # The estimations are reset before the sampling rather than after,
            # so that a sampling interrupted by an error
            # does not leak its samples into the estimations at the next input point.
            # The estimations of all the statistic functions are reset,
            # as the sampling runs the callbacks of all of them
            # and the returned data are cached for all of them at this input point.
            for estimation in estimations:
                estimation.reset()

            new_callbacks = new_callbacks + estimations

        doe_algo_settings.callbacks = list(original_callbacks) + new_callbacks
        doe_algo_settings.use_database = not estimate_statistics_iteratively
        doe_algo_settings.eval_jac = compute_jacobian
        doe_algo_settings.evaluate_observable_jacobian = compute_jacobian
        try:
            with LoggingContext(logging.getLogger("gemseo"), level=logging.ERROR):
                self.__doe_algo.execute(problem, settings=doe_algo_settings)
        finally:
            # Otherwise, an error would leave the callbacks attached
            # and the next sampling would call them twice per sample.
            doe_algo_settings.callbacks = original_callbacks

        if self.__samples_directory_path:
            main_problem = self.problem
            iteration = main_problem.evaluation_counter.current + 1
            dataset = problem.to_dataset(f"Iteration {iteration}")
            dataset.misc.update(
                main_problem.input_space.convert_array_to_dict(input_data)
            )
            to_pickle(dataset, self.__samples_directory_path / f"{iteration}.pkl")
