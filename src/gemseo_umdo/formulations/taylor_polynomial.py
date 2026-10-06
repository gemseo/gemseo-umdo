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
r"""U-MDO formulation based on Taylor polynomials.

[TaylorPolynomial][gemseo_umdo.formulations.taylor_polynomial.TaylorPolynomial] is a
[BaseUMDOFormulation][gemseo_umdo.formulations.base_umdo_formulation.BaseUMDOFormulation]
estimating the statistics with first- or second-order Taylor polynomials
around the expectation of the uncertain variables:

$$f(x,U)\approx f(x,\mu) + (U-\mu)f'(x,\mu).$$

E.g.

$$\mathbb{E}[f(x,U)]\approx
\frac{1}{N}\sum_{i=1}^N f\left(x,U^{(i)}\right)$$

or

$$\mathbb{V}[f(x,U)]\approx \sigma^2f'(x,\mu)$$

where $U$ is normally distributed
with mean $\mu$ and variance $\sigma^2$.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.util.pydantic import create_model

from gemseo_umdo.formulations._functions.hessian_function import HessianFunction
from gemseo_umdo.formulations._functions.statistic_function_for_taylor_polynomial import (  # noqa: E501
    StatisticFunctionForTaylorPolynomial,
)
from gemseo_umdo.formulations._statistics.taylor_polynomial.factory import (  # noqa: E501
    TaylorPolynomialEstimatorFactory,
)
from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation
from gemseo_umdo.formulations.taylor_polynomial_settings import (
    TaylorPolynomial_Settings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.discipline import Discipline
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.optimization import OptimizationProblem
    from gemseo.space import RandomSpace


class TaylorPolynomial(BaseUMDOFormulation):
    """U-MDO formulation based on Taylor polynomials."""

    settings_class: ClassVar[type[TaylorPolynomial_Settings]] = (
        TaylorPolynomial_Settings
    )

    _USE_AUXILIARY_MDO_FORMULATION: ClassVar[bool] = True

    __hessian_fd_problem: EvaluationProblem | None
    """The problem related to the approximation of the Hessian if any."""

    _STATISTIC_FACTORY: ClassVar[TaylorPolynomialEstimatorFactory] = (
        TaylorPolynomialEstimatorFactory()
    )

    _STATISTIC_FUNCTION_CLASS: ClassVar[
        type[StatisticFunctionForTaylorPolynomial] | None
    ] = StatisticFunctionForTaylorPolynomial

    def __init__(  # noqa: D107
        self,
        problem: OptimizationProblem,
        disciplines: Sequence[Discipline],
        settings: TaylorPolynomial_Settings | None = None,
        *,
        uncertain_space: RandomSpace,
        mdo_formulation_settings: BaseFormulationSettings | None = None,
    ) -> None:
        settings = create_model(self.settings_class, settings_model=settings)
        super().__init__(
            problem,
            disciplines,
            settings=settings,
            uncertain_space=uncertain_space,
            mdo_formulation_settings=mdo_formulation_settings,
        )
        self.__hessian_fd_problem = None
        if settings.second_order:
            self.__hessian_fd_problem = EvaluationProblem(self.uncertain_space)

        self._auxiliary_mdo_formulation.problem.differentiation_method = (
            settings.differentiation_method
        )
        self.problem.differentiation_method = (
            self.problem.DifferentiationMethod.FINITE_DIFFERENCES
        )

    @property
    def hessian_fd_problem(self) -> EvaluationProblem | None:
        """The problem related to the approximation of the Hessian."""
        return self.__hessian_fd_problem

    @property
    def second_order(self) -> bool:
        """Whether to use a second order approximation."""
        return self._settings.second_order

    def _post_add_mdo_observable(self) -> None:
        if self.__hessian_fd_problem is not None:
            self.__hessian_fd_problem.add_observable(
                HessianFunction(self._auxiliary_mdo_formulation.problem.observables[-1])
            )
