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
r"""PCE-based U-MDO formulation.

[PCE][gemseo_umdo.formulations.pce.PCE] is an
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
estimating the statistics from the coefficients of a polynomial chaos expansion (PCE).

E.g.

$$\mathbb{E}[f(x,U)] \approx \alpha_0$$

or

$$\mathbb{V}[f(x,U)] \approx \sum_{1<i\leq P}\alpha_i^2$$

where $(\alpha_i)_{1\leq i \leq N}$ are the coefficients of the PCE

$$\hat{f}_x(U)=\alpha_0 + \sum_{1<i\leq P}\alpha_i\Phi_i(U)$$

built at $x$ over the uncertain space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from gemseo_umdo.formulations._functions.statistic_function_for_pce import (
    StatisticFunctionForPCE,
)
from gemseo_umdo.formulations._statistics.pce.factory import PCEEstimatorFactory
from gemseo_umdo.formulations.surrogate import Surrogate

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation


class PCE(Surrogate):
    """PCE-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the
        [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html).
        for more information about the available DOE algorithm names and options.
    """

    _STATISTIC_FACTORY: ClassVar[PCEEstimatorFactory] = PCEEstimatorFactory()

    _STATISTIC_FUNCTION_CLASS: ClassVar[type[StatisticFunctionForPCE] | None] = (
        StatisticFunctionForPCE
    )

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        objective_statistic_parameters: Mapping[str, Any] = READ_ONLY_EMPTY_DICT,
        maximize_objective: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        doe_algo: str = "OT_OPT_LHS",
        doe_algo_options: Mapping[str, Any] = READ_ONLY_EMPTY_DICT,
        doe_n_samples: int | None = None,
        pce_options: Mapping[str, Any] = READ_ONLY_EMPTY_DICT,
        quality_name: str = "R2Measure",
        quality_threshold: float | Mapping[str, float | Iterable[float]] = 0.9,
        quality_cv_compute: bool = True,
        quality_cv_n_folds: int = 5,
        quality_cv_randomize: bool = True,
        quality_cv_seed: int | None = None,
        quality_cv_threshold: float | Mapping[str, float | Iterable[float]] = 0.8,
        mdo_formulation_options: Mapping[str, Any] = READ_ONLY_EMPTY_DICT,
        **options: Any,
    ) -> None:
        """
        Args:
            pce_options: The options of the
                [PCERegressor][gemseo.mlearning.regression.algos.pce.PCERegressor].
        """  # noqa: D205 D212 D415
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            objective_statistic_parameters=objective_statistic_parameters,
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
            mdo_formulation_options=mdo_formulation_options,
            doe_algo=doe_algo,
            doe_algo_options=doe_algo_options,
            doe_n_samples=doe_n_samples,
            regressor_name="PCERegressor",
            regressor_options=pce_options,
            quality_name=quality_name,
            quality_threshold=quality_threshold,
            quality_cv_compute=quality_cv_compute,
            quality_cv_n_folds=quality_cv_n_folds,
            quality_cv_randomize=quality_cv_randomize,
            quality_cv_seed=quality_cv_seed,
            quality_cv_threshold=quality_cv_threshold,
            **options,
        )
