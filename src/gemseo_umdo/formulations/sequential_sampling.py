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
r"""Sequential sampling for multidisciplinary design problems under uncertainty.

:class:`.SequentialSampling` is an :class:`.UMDOFormulation`
estimating the statistics with sequential (quasi) Monte Carlo techniques.

E.g.
:math:`\mathbb{E}[f(x,U)] \approx \frac{1}{N}\sum_{i=1}^N f\left(x,U^{(i)}\right)`
or
:math:`\mathbb{V}[f(x,U)] \approx
\frac{1}{N}\sum_{i=1}^N \left(f\left(x,U^{(i)}\right)-
\frac{1}{N}\sum_{j=1}^N f\left(x,U^{(j)}\right)\right)^2`
where :math:`U` is normally distributed
with mean :math:`\mu` and unit variance :math:`\sigma`
and :math:`U^{(k,1)},\ldots,U^{(k,N_k)}` are :math:`N_k` realizations of :math:`U`
obtained at the $k$-th iteration of the optimization loop
with an optimized Latin hypercube sampling technique.
"""
from __future__ import annotations

import logging
from typing import Any
from typing import Mapping
from typing import Sequence

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.formulation import MDOFormulation

from gemseo_umdo.formulations.sampling import Sampling

LOGGER = logging.getLogger(__name__)


class SequentialSampling(Sampling):
    """Sequential sampling-based robust MDO formulation."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: MDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        n_samples: int,
        initial_n_samples: int = 1,
        n_samples_increment: int = 1,
        objective_statistic_parameters: Mapping[str, Any] | None = None,
        maximize_objective: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        algo: str = "OT_OPT_LHS",
        algo_options: Mapping[str, Any] | None = None,
        seed: int = 1,
        **options: Any,
    ) -> None:
        """
        Args:
            initial_n_samples: The initial sampling size.
            n_samples_increment: The increment of the sampling size.
        """  # noqa: D205 D212 D415
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            initial_n_samples,
            objective_statistic_parameters=objective_statistic_parameters,
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
            algo=algo,
            algo_options=algo_options,
            seed=seed,
            **options,
        )
        self.__final_n_samples = n_samples
        self.__n_samples_increment = n_samples_increment

    def compute_samples(self, problem: OptimizationProblem) -> None:  # noqa: D102
        super().compute_samples(problem)
        if self._n_samples < self.__final_n_samples:
            self._n_samples = min(
                self.__final_n_samples, self._n_samples + self.__n_samples_increment
            )
