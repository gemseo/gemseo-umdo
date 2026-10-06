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
from typing import Any
from typing import ClassVar

from gemseo.optimization import OptimizationProblem
from gemseo.core.function.array_function import ArrayFunction
from gemseo.util.constant import read_only_empty_dict

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

    from gemseo.space import DesignSpace
    from gemseo.space import RandomSpace
    from gemseo.discipline import Discipline
    from gemseo.formulation.core.base_mdo import BaseMDOFormulation
    from gemseo.util.typing import StrKeyMapping


class TaylorPolynomial(BaseUMDOFormulation):
    """U-MDO formulation based on Taylor polynomials."""

    settings_class: ClassVar[type[TaylorPolynomial_Settings]] = TaylorPolynomial_Settings

    _USE_AUXILIARY_MDO_FORMULATION: ClassVar[bool] = True

    __hessian_fd_problem: OptimizationProblem | None
    """The problem related to the approximation of the Hessian if any."""

    _STATISTIC_FACTORY: ClassVar[TaylorPolynomialEstimatorFactory] = (
        TaylorPolynomialEstimatorFactory()
    )

    _STATISTIC_FUNCTION_CLASS: ClassVar[
        type[StatisticFunctionForTaylorPolynomial] | None
    ] = StatisticFunctionForTaylorPolynomial

    # TODO(bump-gemseo): pass the problem first, e.g. OptimizationProblem(design_space), then set its objective; the loose settings go into settings=<Formulation>_Settings(...)  # noqa: E501
    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: RandomSpace,
        objective_statistic_name: str,
        settings: TaylorPolynomial_Settings,
        minimize_objective: bool = True,
        objective_statistic_parameters: StrKeyMapping = read_only_empty_dict,
        mdo_formulation_settings: StrKeyMapping = read_only_empty_dict) -> None:
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

        self.__hessian_fd_problem = None
        problem = self._auxiliary_mdo_formulation.optimization_problem
        if settings.second_order:
            self.__hessian_fd_problem = OptimizationProblem(self.uncertain_space)
            self.__hessian_fd_problem.objective = HessianFunction(problem.objective)

        problem.differentiation_method = settings.differentiation_method
        problem.design_space = problem.design_space.to_design_space()
        self.problem.differentiation_method = (
            self.problem.ApproximationMode.FINITE_DIFFERENCES
        )
        self.problem.fd_step = 1e-6

    @property
    def hessian_fd_problem(self) -> OptimizationProblem | None:
        """The problem related to the approximation of the Hessian."""
        return self.__hessian_fd_problem

    @property
    def second_order(self) -> bool:
        """Whether to use a second order approximation."""
        return self._settings.second_order

    def add_constraint(  # noqa: D102
        self,
        output_name: str | Sequence[str],
        statistic_name: str,
        constraint_type: ArrayFunction.ConstraintType = ArrayFunction.ConstraintType.INEQ,
        constraint_name: str = "",
        value: float = 0.0,
        positive: bool = False,
        **statistic_parameters: Any,
    ) -> None:
        super().add_constraint(
            output_name,
            statistic_name,
            constraint_type=constraint_type,
            constraint_name=constraint_name,
            value=value,
            positive=positive,
            **statistic_parameters,
        )
        if self.hessian_fd_problem is not None:
            self.hessian_fd_problem.add_observable(
                HessianFunction(
                    self._auxiliary_mdo_formulation.optimization_problem.observables[-1]
                )
            )

    def add_observable(  # noqa: D102
        self,
        output_names: Sequence[str],
        statistic_name: str,
        observable_name: Sequence[str] = "",
        discipline: Discipline | None = None,
        **statistic_parameters: Any,
    ) -> None:
        super().add_observable(
            output_names,
            statistic_name,
            observable_name=observable_name,
            discipline=discipline,
            **statistic_parameters,
        )
        if self.hessian_fd_problem is not None:
            self.hessian_fd_problem.add_observable(
                HessianFunction(
                    self._auxiliary_mdo_formulation.optimization_problem.observables[-1]
                ),
            )
