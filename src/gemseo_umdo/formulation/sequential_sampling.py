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

[SequentialSampling][gemseo_umdo.formulation.sequential_sampling.SequentialSampling]
is a
[BaseUMDOFormulation][gemseo_umdo.formulation.core.base.BaseUMDOFormulation]
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

from gemseo.util.hashable_ndarray import HashableNdarray
from gemseo.util.pydantic import create_model

from gemseo_umdo.formulation.sampling import Sampling
from gemseo_umdo.formulation.sequential_sampling_settings import (
    SequentialSampling_Settings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from gemseo.core.problem.evaluation import EvaluationProblem
    from gemseo.discipline import Discipline
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import RandomSpace
    from gemseo.util.typing import RealArray


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

    __n_samples: int
    """The number of samples of the next sampling.

    It starts at `initial_n_samples`
    and increases up to the `n_samples` of the DOE settings,
    which are left unchanged.
    """

    __last_input_point: HashableNdarray | None
    """The last sampled input point, if any.

    The number of samples is increased
    only when sampling an input point different from this one,
    so that sampling the same input point again,
    e.g. to estimate the Jacobians after the statistics,
    uses the same number of samples.
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
        self.__n_samples = settings.initial_n_samples
        self.__last_input_point = None
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
        # The input point is copied
        # so that a change of the input array during or after this call
        # does not change the key.
        input_point = HashableNdarray(input_data, copy=True)
        last_input_point = self.__last_input_point
        if last_input_point is not None and input_point != last_input_point:
            n_samples = self.__n_samples
            self.__n_samples = min(
                self._settings.doe_algo_settings.n_samples,
                n_samples + self._settings.n_samples_increment(n_samples),
            )

        self.__last_input_point = input_point
        super().compute_samples(problem, input_data, compute_jacobian=compute_jacobian)

    def _get_doe_algo_settings_update(self) -> dict[str, Any]:  # noqa: D102
        return {"n_samples": self.__n_samples}
