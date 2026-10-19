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
r"""Control variate-based U-MDO formulation.

[ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate] is a
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
estimating the statistics with first-order Taylor polynomials as control variates:

$$\tilde{f}(x,u)=f(x,\mu) + (u-\mu)\frac{\partial f(x,\mu)}{\partial u}$$

where $u$ is a realization of the random variable $U$ and $\mu=\mathbb{E}[U]$.

The expectation $\mathbb{E}[f(x,U)]$ can be approximated
by the control variate estimator

$$\frac{1}{N}\sum_{i=1}^N f\left(x,U^{(i)}\right)
+\alpha_N\left(\frac{1}{N}\sum_{j=1}^N \tilde{f}\left(x,U^{(j)}\right)-f(x,\mu)\right)$$

where $\alpha_N$ is the empirical estimator
of $\frac{\text{cov}\left[f(x,U),\tilde{f}(x,u)\right]}
{\mathbb{V}\left[f(x,U)\right]}$
and $U^{(1)},\ldots,U^{(N)}$ are $N$ independent realizations of $U$.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.doe.factory import DOELibraryFactory
from gemseo.util.logging import LoggingContext
from gemseo.util.pydantic import create_model

from gemseo_umdo.formulations._functions.statistic_function_for_control_variate import (
    StatisticFunctionForControlVariate,
)
from gemseo_umdo.formulations._statistics.control_variate.factory import (  # noqa: E501
    ControlVariateEstimatorFactory,
)
from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation
from gemseo_umdo.formulations.control_variate_settings import ControlVariate_Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.problem.evaluation import EvaluationProblem
    from gemseo.discipline import Discipline
    from gemseo.doe.core.base_doe_library import BaseDOELibrary
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import RandomSpace


class ControlVariate(BaseUMDOFormulation):
    """Control variate-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the documentation of the [DOE algorithms][gemseo.doe]
        for more information about the available DOE algorithms and their settings.
    """

    settings_class: ClassVar[type[ControlVariate_Settings]] = ControlVariate_Settings

    _USE_AUXILIARY_MDO_FORMULATION: ClassVar[bool] = True

    __doe_algo: BaseDOELibrary
    """The DOE algorithm to sample the original functions of the uncertain problem."""

    __doe_algo_for_regressor: BaseDOELibrary
    """The DOE algorithm to sample the regressor."""

    _STATISTIC_FACTORY: ClassVar[ControlVariateEstimatorFactory] = (
        ControlVariateEstimatorFactory()
    )

    _STATISTIC_FUNCTION_CLASS: ClassVar[
        type[StatisticFunctionForControlVariate] | None
    ] = StatisticFunctionForControlVariate

    def __init__(  # noqa: D107
        self,
        problem: OptimizationProblem,
        disciplines: Sequence[Discipline],
        settings: ControlVariate_Settings | None = None,
        *,
        uncertain_space: RandomSpace,
        mdo_formulation_settings: BaseFormulationSettings | None = None,
    ) -> None:
        settings = create_model(self.settings_class, settings_model=settings)
        algo_name = settings.doe_algo_settings.target_class_name
        self.__doe_algo = DOELibraryFactory().create(algo_name)
        self.__doe_algo_for_regressor = DOELibraryFactory().create(
            settings.regressor_doe_algo_settings.target_class_name
        )
        super().__init__(
            problem,
            disciplines,
            settings=settings,
            uncertain_space=uncertain_space,
            mdo_formulation_settings=mdo_formulation_settings,
        )
        self.name = (
            f"{self.__class__.__name__}"
            f"[{self._mdo_formulation.__class__.__name__}; "
            f"{algo_name}({settings.n_samples})]"
        )

    @property
    def doe_algo(self) -> BaseDOELibrary:
        """The DOE library configured with an algorithm."""
        return self.__doe_algo

    def compute_samples(
        self, problem: EvaluationProblem, create_training_dataset: bool = False
    ) -> None:
        """Evaluate the functions of a problem with a DOE algorithm.

        Args:
            problem: The problem.
            create_training_dataset: Whether to evaluate the functions
                to create a training dataset.
        """
        if create_training_dataset:
            doe_algo = self.__doe_algo_for_regressor
            settings_model = self._settings.regressor_doe_algo_settings
        else:
            doe_algo = self.__doe_algo
            settings_model = self._settings.doe_algo_settings

        with LoggingContext(logging.getLogger("gemseo")):
            doe_algo.execute(problem, settings=settings_model)
