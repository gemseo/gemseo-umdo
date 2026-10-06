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
r"""Sequential sampling-based U-MDO formulation.

[SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling]
is a
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
estimating the statistics with sequential (quasi) Monte Carlo techniques.

E.g.

$$\mathbb{E}[f(x,U)] \approx \frac{1}{N_k}\sum_{i=1}^{N_k} f\left(x,U^{(k,i)}\right)$$

or

$$\mathbb{V}[f(x,U)] \approx
\frac{1}{N_k-1}\sum_{i=1}^{N_k} \left(f\left(x,U^{(k,i)}\right)-
\frac{1}{N_k}\sum_{j=1}^{N_k} f\left(x,U^{(k,j)}\right)\right)^2$$

where $U$ is normally distributed
with mean $\mu$ and variance $\sigma^2$
and $U^{(k,1)},\ldots,U^{(k,N_k)}$ are $N_k$ realizations of $U$
obtained at the $k$-th iteration of the optimization loop
with an optimized Latin hypercube sampling technique.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.util.constant import read_only_empty_dict

from gemseo_umdo.formulations.sampling import Sampling
from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.space import DesignSpace
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import RandomSpace
    from gemseo.discipline import Discipline
    from gemseo.formulation.core.base_mdo import BaseMDOFormulation
    from gemseo.util.typing import RealArray
    from gemseo.util.typing import StrKeyMapping


class SequentialSampling(Sampling):
    """Sequential sampling-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the
        [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html).
        for more information about the available DOE algorithm names and options.
    """

    settings_class: ClassVar[type[SequentialSampling_Settings]] = SequentialSampling_Settings

    __final_n_samples: int
    """The maximum number of samples when evaluating the U-MDO formulation."""

    # TODO(bump-gemseo): pass the problem first, e.g. OptimizationProblem(design_space), then set its objective; the loose settings go into settings=<Formulation>_Settings(...)  # noqa: E501
    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: RandomSpace,
        objective_statistic_name: str,
        settings: SequentialSampling_Settings,
        minimize_objective: bool = True,
        objective_statistic_parameters: StrKeyMapping = read_only_empty_dict,
        mdo_formulation_settings: StrKeyMapping = read_only_empty_dict) -> None:
        self.__final_n_samples = settings.doe_algo_settings.n_samples
        settings.doe_algo_settings.n_samples = settings.initial_n_samples
        # TODO(bump-gemseo): pass the problem first, e.g. OptimizationProblem(design_space), then set its objective; the loose settings go into settings=<Formulation>_Settings(...)  # noqa: E501
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            settings,
            minimize_objective=minimize_objective,
            objective_statistic_parameters=objective_statistic_parameters,
            mdo_formulation_settings=mdo_formulation_settings,
        )

    def compute_samples(  # noqa: D102
        self,
        problem: OptimizationProblem,
        input_data: RealArray,
        compute_jacobian: bool = False,
    ) -> None:
        super().compute_samples(problem, input_data, compute_jacobian=compute_jacobian)
        doe_algo_settings = self._settings.doe_algo_settings
        if (n_samples := doe_algo_settings.n_samples) < self.__final_n_samples:
            doe_algo_settings.n_samples = min(
                self.__final_n_samples,
                n_samples + self._settings.n_samples_increment(n_samples),
            )
