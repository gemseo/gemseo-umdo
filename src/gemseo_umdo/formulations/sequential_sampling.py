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
from typing import Final

from gemseo.util.hashable_ndarray import HashableNdarray
from gemseo.util.pydantic import create_model

from gemseo_umdo.formulations.sampling import Sampling
from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.problem.evaluation import EvaluationProblem
    from gemseo.discipline import Discipline
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import RandomSpace
    from gemseo.util.typing import RealArray


_max_n_input_points: Final[int] = 10
"""The maximum number of input points whose numbers of samples are kept.

Only the most recent input points sampled without the Jacobians are kept,
so that the memory does not grow with the number of input points,
e.g. with a long derivative-free optimization.
"""


class SequentialSampling(Sampling):
    """Sequential sampling-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the documentation of the [DOE algorithms][gemseo.doe]
        for more information about the available DOE algorithms and their settings.
    """

    settings_class: ClassVar[type[SequentialSampling_Settings]] = (
        SequentialSampling_Settings
    )

    __final_n_samples: int
    """The maximum number of samples when evaluating the U-MDO formulation."""

    __input_data_to_n_samples: dict[HashableNdarray, int]
    """The input points sampled without the Jacobians and their numbers of samples.

    At the end of each execution of the U-scenario,
    only the input points whose statistics are still cached are kept,
    e.g. the last one,
    so that another input point sampled by a previous execution
    follows the schedule of the numbers of samples,
    an input point is removed when its statistics are estimated again,
    so that this new estimation follows this schedule too,
    and it is limited to the `_max_n_input_points` most recent input points,
    so that an older one follows this schedule too.
    It does not depend on `input_data_to_output_data`,
    which may no longer contain an input point
    when its Jacobians are requested,
    e.g. after the evaluation of the functions at another input point.
    """

    def __init__(  # noqa: D107
        self,
        problem: OptimizationProblem,
        disciplines: Sequence[Discipline],
        settings: SequentialSampling_Settings | None = None,
        *,
        uncertain_space: RandomSpace,
        mdo_formulation_settings: BaseFormulationSettings | None = None,
    ) -> None:
        settings = create_model(self.settings_class, settings_model=settings)
        self.__final_n_samples = settings.doe_algo_settings.n_samples
        settings.doe_algo_settings.n_samples = settings.initial_n_samples
        self.__input_data_to_n_samples = {}
        super().__init__(
            problem,
            disciplines,
            settings=settings,
            uncertain_space=uncertain_space,
            mdo_formulation_settings=mdo_formulation_settings,
        )

    def compute_samples(  # noqa: D102
        self,
        problem: EvaluationProblem,
        input_data: RealArray,
        compute_jacobian: bool = False,
    ) -> None:
        doe_algo_settings = self._settings.doe_algo_settings
        n_samples = doe_algo_settings.n_samples
        # The key is a copy of the input point
        # so that a change of the input array during or after this call
        # does not change the key.
        key = HashableNdarray(input_data, copy=True)
        # The resampling of an input point sampled without the Jacobians
        # to estimate them after the statistics
        # uses the same number of samples and does not advance the schedule,
        # so that the statistics and their Jacobians at this input point
        # are estimated with the same number of samples.
        input_data_to_n_samples = self.__input_data_to_n_samples
        last_n_samples = input_data_to_n_samples.get(key) if compute_jacobian else None

        if last_n_samples is not None:
            doe_algo_settings.n_samples = last_n_samples

        try:
            super().compute_samples(
                problem, input_data, compute_jacobian=compute_jacobian
            )
        finally:
            doe_algo_settings.n_samples = n_samples

        if last_n_samples is not None:
            del input_data_to_n_samples[key]
            return

        if not compute_jacobian:
            # The most recent input point is the last one of the dictionary.
            input_data_to_n_samples.pop(key, None)
            input_data_to_n_samples[key] = n_samples
            if len(input_data_to_n_samples) > _max_n_input_points:
                del input_data_to_n_samples[next(iter(input_data_to_n_samples))]

        if n_samples < self.__final_n_samples:
            doe_algo_settings.n_samples = min(
                self.__final_n_samples,
                n_samples + self._settings.n_samples_increment(n_samples),
            )

    def _forget_input_point(self, input_data: RealArray) -> None:
        # A new estimation of the statistics at this input point
        # follows the schedule of the numbers of samples,
        # and so does the estimation of their Jacobians from the same sampling.
        self.__input_data_to_n_samples.pop(HashableNdarray(input_data), None)

    def reset_execution_state(self) -> None:  # noqa: D102
        super().reset_execution_state()
        # The statistics still cached are kept,
        # so that evaluating them after the execution,
        # e.g. by a post-optimal analysis,
        # returns the values of the execution without a new sampling.
        # The numbers of samples of the input points whose statistics are cached
        # are kept too,
        # so that their Jacobians requested after the execution
        # are estimated with the same number of samples as these statistics,
        # and so from the same samples with a seeded DOE algorithm.
        # The numbers of samples of the other input points are forgotten,
        # so that their new samplings follow the schedule of the numbers of samples.
        input_data_to_output_data = self.input_data_to_output_data
        self.__input_data_to_n_samples = {
            input_data: n_samples
            for input_data, n_samples in self.__input_data_to_n_samples.items()
            if input_data in input_data_to_output_data
        }
