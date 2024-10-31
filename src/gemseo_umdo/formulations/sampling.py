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
from typing import Any
from typing import ClassVar

from gemseo import to_pickle
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.logging_tools import LoggingContext
from gemseo.utils.seeder import SEED

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
from gemseo_umdo.formulations.sampling_settings import SamplingSettings

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

    Settings: ClassVar[type[SamplingSettings]] = SamplingSettings

    _estimate_statistics_iteratively: bool
    """Whether to estimate the statistics iteratively."""

    __doe_algo: BaseDOELibrary
    """The DOE library to execute the DOE algorithm."""

    __doe_algo_options: dict[str, Any]
    """The options of the DOE algorithm."""

    __n_samples: int | None
    """The number of samples, if defined."""

    __samples_directory_path: str | Path
    """The path to the directory where the samples are saved."""

    callbacks: list[CallbackType]
    """The callback functions for the DOE algorithm when computing the output data."""

    jacobian_callbacks: list[CallbackType]
    """The callback functions for the DOE algorithm when computing the Jacobian."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        n_samples: int | None = None,
        objective_statistic_parameters: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        algo: str = "OT_OPT_LHS",
        algo_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        seed: int = SEED,
        estimate_statistics_iteratively: bool = True,
        samples_directory_path: str | Path = "",
        mdo_formulation_settings: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        **settings: Any,
    ) -> None:
        """
        Args:
            n_samples: The number of samples to be generated by the DOE algorithm.
                If `None`,
                the DOE algorithm uses no `n_samples` argument
                but potentially a mandatory argument to be defined in `algo_options`
                (e.g. `samples` for the `CustomDOE` algorithm).
            algo: The name of the DOE algorithm.
            algo_options: The options of the DOE algorithm.
            seed: The seed for reproducibility.
            estimate_statistics_iteratively: Whether to estimate
                the statistics iteratively for memory reasons.
                This argument is ignored when `samples_directory_path` is defined;
                in this case, the statistics are not estimated iteratively.
            samples_directory_path: The path to a new directory
                where the samples stored as :class:`.IODataset` objects will be saved
                (one object per file, one file per iteration).
                This directory must not exist; it will be created by the formulation.
                If empty, do not save the samples.

        Raises:
            ValueError: When `n_samples` is `None`,
                whereas it is required by the DOE algorithm.
        """  # noqa: D205 D212 D415
        self.callbacks = []
        self.jacobian_callbacks = []
        self.input_data_to_output_samples = {}
        if samples_directory_path:
            self.__samples_directory_path = Path(samples_directory_path)
            self.__samples_directory_path.mkdir()
            estimate_statistics_iteratively = False
        else:
            self.__samples_directory_path = ""

        self._estimate_statistics_iteratively = estimate_statistics_iteratively
        if estimate_statistics_iteratively:
            self._statistic_factory = IterativeSamplingEstimatorFactory()
            self._statistic_function_class = StatisticFunctionForIterativeSampling
        else:
            self._statistic_factory = SamplingEstimatorFactory()
            self._statistic_function_class = StatisticFunctionForStandardSampling

        self.__doe_algo = DOELibraryFactory().create(algo)
        self.__doe_algo_options = dict(algo_options)
        self.__doe_algo_options["use_database"] = not estimate_statistics_iteratively
        model_fields = self.__doe_algo.ALGORITHM_INFOS[algo].Settings.model_fields
        if "n_samples" in model_fields:
            if n_samples is None:
                msg = "Sampling: n_samples is required."
                raise ValueError(msg)
            self.__doe_algo_options["n_samples"] = n_samples

        if "seed" in model_fields:
            self.__doe_algo_options["seed"] = seed

        self.__n_samples = n_samples
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            objective_statistic_parameters=objective_statistic_parameters,
            mdo_formulation_settings=mdo_formulation_settings,
            **settings,
        )
        mdo_formulation = self._mdo_formulation.__class__.__name__
        formulation = self.__class__.__name__
        self.name = f"{formulation}[{mdo_formulation}; {algo}({n_samples})]"

    @property
    def _n_samples(self) -> int | None:
        """The number of samples, if defined."""
        return self.__doe_algo_options.get("n_samples")

    @_n_samples.setter
    def _n_samples(self, value: int) -> None:
        self.__doe_algo_options["n_samples"] = value

    @property
    def _algo(self) -> BaseDOELibrary:
        """The DOE algorithm."""
        return self.__doe_algo

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
        callbacks_1 = list(self.__doe_algo_options.get("callbacks", []))
        callbacks_2 = self.jacobian_callbacks if compute_jacobian else self.callbacks
        self.__doe_algo_options["callbacks"] = callbacks_1 + callbacks_2
        with LoggingContext(logging.getLogger("gemseo")):
            self.__doe_algo.execute(
                problem,
                eval_jac=compute_jacobian,
                eval_obs_jac=compute_jacobian,
                **self.__doe_algo_options,
            )

        if self.__samples_directory_path:
            main_problem = self.optimization_problem
            iteration = main_problem.evaluation_counter.current + 1
            dataset = problem.to_dataset(f"Iteration {iteration}", opt_naming=False)
            dataset.misc.update(
                main_problem.design_space.convert_array_to_dict(input_data)
            )
            to_pickle(dataset, self.__samples_directory_path / f"{iteration}.pkl")
