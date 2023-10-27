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
r"""Sampling for multidisciplinary design problems under uncertainty.

[Sampling][gemseo_umdo.formulations.sampling.Sampling] is an
[UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation]
estimating the statistics with (quasi) Monte Carlo techniques.

E.g.
$\mathbb{E}[f(x,U)] \approx \frac{1}{N}\sum_{i=1}^N f\left(x,U^{(i)}\right)$
or
$\mathbb{V}[f(x,U)] \approx
\frac{1}{N}\sum_{i=1}^N \left(f\left(x,U^{(i)}\right)-
\frac{1}{N}\sum_{j=1}^N f\left(x,U^{(j)}\right)\right)^2$
where $U$ is normally distributed
with mean $\mu$ and variance $\sigma^2$
and $U^{(1)},\ldots,U^{(N)}$ are $N$ realizations of $U$
obtained with an optimized Latin hypercube sampling technique.
"""
from __future__ import annotations

from typing import Any
from typing import Mapping
from typing import Sequence

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.doe_library import DOELibrary
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.formulation import MDOFormulation
from gemseo.utils.logging_tools import LoggingContext

from gemseo_umdo.formulations.formulation import UMDOFormulation
from gemseo_umdo.formulations.functions.statistic_function_for_iterative_sampling import (
    StatisticFunctionForIterativeSampling,
)
from gemseo_umdo.formulations.functions.statistic_function_for_standard_sampling import (
    StatisticFunctionForStandardSampling,
)
from gemseo_umdo.formulations.statistics.iterative_sampling.sampling_estimator_factory import (
    SamplingEstimatorFactory as IterativeSamplingEstimatorFactory,
)
from gemseo_umdo.formulations.statistics.sampling.sampling_estimator_factory import (
    SamplingEstimatorFactory,
)


class Sampling(UMDOFormulation):
    """Sampling-based robust MDO formulation."""

    _estimate_statistics_iteratively: bool
    """Whether to estimate the statistics iteratively."""

    __doe_algo: DOELibrary
    """The DOE library to execute the DOE algorithm."""

    __doe_algo_options: dict[str, Any]
    """The options of the DOE algorithm."""

    __n_samples: int
    """The number of samples."""

    __seed: int
    """The seed for reproducibility."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: MDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        n_samples: int,
        objective_statistic_parameters: Mapping[str, Any] | None = None,
        maximize_objective: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        algo: str = OpenTURNS.OT_LHSO,
        algo_options: Mapping[str, Any] | None = None,
        seed: int = 1,
        estimate_statistics_iteratively: bool = True,
        **options: Any,
    ) -> None:
        """
        Args:
            n_samples: The number of samples, i.e. the size of the DOE.
            algo: The name of the DOE algorithm.
            algo_options: The options of the DOE algorithm.
            seed: The seed for reproducibility.
            estimate_statistics_iteratively: Whether to estimate
                the statistics iteratively.
        """  # noqa: D205 D212 D415
        self._estimate_statistics_iteratively = estimate_statistics_iteratively
        if estimate_statistics_iteratively:
            self._statistic_factory = IterativeSamplingEstimatorFactory()
            self._statistic_function_class = StatisticFunctionForIterativeSampling
        else:
            self._statistic_factory = SamplingEstimatorFactory()
            self._statistic_function_class = StatisticFunctionForStandardSampling

        self.__doe_algo = DOEFactory().create(algo)
        self.__doe_algo_options = algo_options or {}
        self.__doe_algo_options[
            DOELibrary.USE_DATABASE_OPTION
        ] = not estimate_statistics_iteratively
        self.__doe_algo_options[DOELibrary.N_SAMPLES] = n_samples
        self.__n_samples = n_samples
        self.__seed = seed
        self._estimators = []
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            objective_statistic_options=objective_statistic_parameters,
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
            **options,
        )
        mdo_formulation = self._mdo_formulation.__class__.__name__
        formulation = self.__class__.__name__
        self.name = f"{formulation}[{mdo_formulation}; {algo}({n_samples})]"

    @property
    def _n_samples(self) -> int:
        """The number of samples."""
        return self.__doe_algo_options[DOELibrary.N_SAMPLES]

    @_n_samples.setter
    def _n_samples(self, value: int) -> None:
        self.__doe_algo_options[DOELibrary.N_SAMPLES] = value

    @property
    def _algo(self) -> DOELibrary:
        """The DOE algorithm."""
        return self.__doe_algo

    def compute_samples(self, problem: OptimizationProblem) -> None:
        """Evaluate the functions of a problem with a DOE algorithm.

        Args:
            problem: The problem.
        """
        with LoggingContext():
            self.__doe_algo.execute(
                problem, seed=self.__seed, **self.__doe_algo_options
            )
