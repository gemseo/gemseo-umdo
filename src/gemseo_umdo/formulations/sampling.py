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

:class:`~gemseo_umdo.formulations.sampling.Sampling` is an
:class:`~gemseo_umdo.formulations.formulation.UMDOFormulation`
estimating the statistics with (quasi) Monte Carlo techniques.

E.g.
:math:`\mathbb{E}[f(x,U)] \approx \frac{1}{N}\sum_{i=1}^N f\left(x,U^{(i)}\right)`
or
:math:`\mathbb{V}[f(x,U)] \approx
\frac{1}{N}\sum_{i=1}^N \left(f\left(x,U^{(i)}\right)-
\frac{1}{N}\sum_{j=1}^N f\left(x,U^{(j)}\right)\right)^2`
where :math:`U` is normally distributed
with mean :math:`\mu` and unit variance :math:`\sigma`
and :math:`U^{(1)},\ldots,U^{(1)}` are :math:`N` realizations of :math:`U`
obtained with an optimized Latin hypercube sampling technique.
"""
from __future__ import annotations

import logging
from typing import Any
from typing import ClassVar
from typing import Mapping
from typing import Sequence

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.doe_lib import DOELibrary
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.formulation import MDOFormulation
from gemseo.utils.logging_tools import LoggingContext
from numpy import ndarray

from gemseo_umdo.estimators.sampling import SamplingEstimatorFactory
from gemseo_umdo.formulations.formulation import UMDOFormulation

LOGGER = logging.getLogger(__name__)


class Sampling(UMDOFormulation):
    """Sampling-based robust MDO formulation."""

    _STATISTIC_FACTORY: ClassVar = SamplingEstimatorFactory()

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
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        algo: str = "OT_OPT_LHS",
        algo_options: Mapping[str, Any] | None = None,
        seed: int = 1,
        **options: Any,
    ) -> None:
        """# noqa: D205 D212 D415
        Args:
            n_samples: The number of samples, i.e. the size of the DOE.
            algo: The name of the DOE algorithm.
            algo_options: The options of the DOE algorithm.
        """
        self.__doe_algo = DOEFactory().create(algo)
        self.__doe_algo_options = algo_options or {}
        self.__doe_algo_options["n_samples"] = n_samples
        self.__n_samples = n_samples
        self.processed_functions = []
        self.__seed = seed
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
        return self.__doe_algo_options["n_samples"]

    @_n_samples.setter
    def _n_samples(self, value: int) -> None:
        self.__doe_algo_options["n_samples"] = value

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
            self.__doe_algo.seed = self.__seed
            self.__doe_algo.execute(
                problem, seed=self.__seed, **self.__doe_algo_options
            )

    class _StatisticFunction(UMDOFormulation._StatisticFunction):
        def _func(self, input_data: ndarray) -> ndarray:
            formulation = self._formulation
            problem = formulation.mdo_formulation.opt_problem
            if self._function_name in formulation._processed_functions:
                formulation._processed_functions = []
                problem.reset()

            database = problem.database
            if not database:
                formulation.update_top_level_disciplines(input_data)
                formulation.compute_samples(problem)

            formulation._processed_functions.append(self._function_name)
            samples, _, _ = database.get_history_array(
                [self._function_name], add_dv=False
            )
            return self._estimate_statistic(samples, **self._statistic_parameters)
