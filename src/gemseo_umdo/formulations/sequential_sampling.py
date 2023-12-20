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

[SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling]
is an [UMDOFormulation][gemseo_umdo.formulations.formulation.UMDOFormulation]
estimating the statistics with sequential (quasi) Monte Carlo techniques.

E.g.
$\mathbb{E}[f(x,U)] \approx \frac{1}{N_k}\sum_{i=1}^{N_k} f\left(x,U^{(k,i)}\right)$
or
$\mathbb{V}[f(x,U)] \approx
\frac{1}{N_k-1}\sum_{i=1}^{N_k} \left(f\left(x,U^{(k,i)}\right)-
\frac{1}{N_k}\sum_{j=1}^{N_k} f\left(x,U^{(k,j)}\right)\right)^2$
where $U$ is normally distributed
with mean $\mu$ and variance $\sigma^2$
and $U^{(k,1)},\ldots,U^{(k,N_k)}$ are $N_k$ realizations of $U$
obtained at the $k$-th iteration of the optimization loop
with an optimized Latin hypercube sampling technique.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.discipline import MDODiscipline

from gemseo_umdo.formulations.sampling import Sampling

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.formulation import MDOFormulation


class SequentialSampling(Sampling):
    """Sequential sampling-based robust MDO formulation."""

    __final_n_samples: int
    """The maximum number of samples when evaluating the U-MDO formulation."""

    __n_samples_increment: int
    """The increment of the sampling size."""

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
