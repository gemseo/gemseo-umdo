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
from typing import ClassVar

from gemseo.mlearning.regression.algos.pce_settings import PCERegressor_Settings
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from gemseo_umdo.formulations._functions.statistic_function_for_pce import (
    StatisticFunctionForPCE,
)
from gemseo_umdo.formulations._statistics.pce.factory import PCEEstimatorFactory
from gemseo_umdo.formulations.pce_settings import PCE_Settings
from gemseo_umdo.formulations.surrogate import Surrogate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
    from gemseo.typing import StrKeyMapping


class PCE(Surrogate):
    """PCE-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the
        [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html).
        for more information about the available DOE algorithm names and options.
    """

    Settings: ClassVar[type[PCE_Settings]] = PCE_Settings

    _STATISTIC_FACTORY: ClassVar[PCEEstimatorFactory] = PCEEstimatorFactory()

    _STATISTIC_FUNCTION_CLASS: ClassVar[type[StatisticFunctionForPCE] | None] = (
        StatisticFunctionForPCE
    )

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        settings_model: PCE_Settings,
        objective_statistic_parameters: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        mdo_formulation_settings: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> None:
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            settings_model,
            objective_statistic_parameters=objective_statistic_parameters,
            mdo_formulation_settings=mdo_formulation_settings,
        )
        regressor_settings = self._settings.regressor_settings.model_dump()
        regressor_settings["probability_space"] = uncertain_space
        self._settings.regressor_settings = PCERegressor_Settings(**regressor_settings)
