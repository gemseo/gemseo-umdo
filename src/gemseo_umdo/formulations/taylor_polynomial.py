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

[TaylorPolynomial][gemseo_umdo.formulations.taylor_polynomial.TaylorPolynomial] is an
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

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from gemseo_umdo.formulations._functions.hessian_function import HessianFunction
from gemseo_umdo.formulations._functions.statistic_function_for_taylor_polynomial import (  # noqa: E501
    StatisticFunctionForTaylorPolynomial,
)
from gemseo_umdo.formulations._statistics.taylor_polynomial.factory import (  # noqa: E501
    TaylorPolynomialEstimatorFactory,
)
from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation


class TaylorPolynomial(BaseUMDOFormulation):
    """U-MDO formulation based on Taylor polynomials."""

    _USE_AUXILIARY_MDO_FORMULATION: ClassVar[bool] = True

    __hessian_fd_problem: OptimizationProblem | None
    """The problem related to the approximation of the Hessian if any."""

    __second_order: bool
    """Whether the formulation uses second-order Taylor polynomials.

    Otherwise first-order Taylor polynomials.
    """

    _STATISTIC_FACTORY: ClassVar[TaylorPolynomialEstimatorFactory] = (
        TaylorPolynomialEstimatorFactory()
    )

    _STATISTIC_FUNCTION_CLASS: ClassVar[
        type[StatisticFunctionForTaylorPolynomial] | None
    ] = StatisticFunctionForTaylorPolynomial

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: BaseMDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        objective_statistic_parameters: Mapping[str, Any] = READ_ONLY_EMPTY_DICT,
        maximize_objective: bool = False,
        differentiation_method: OptimizationProblem.DifferentiationMethod = OptimizationProblem.DifferentiationMethod.USER_GRAD,  # noqa: E501
        second_order: bool = False,
        mdo_formulation_options: Mapping[str, Any] = READ_ONLY_EMPTY_DICT,
        **options: Any,
    ) -> None:  # noqa: D205 D212 D415
        """
        Args:
            differentiation_method: The type of method to compute the gradients.
            second_order: Whether to use second-order Taylor polynomials
                instead of first-order Taylor polynomials.
        """  # noqa: D205 D212 D415
        self.__second_order = second_order
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            mdo_formulation,
            uncertain_space,
            objective_statistic_name,
            objective_statistic_parameters=objective_statistic_parameters,
            maximize_objective=maximize_objective,
            mdo_formulation_options=mdo_formulation_options,
            **options,
        )

        self.__hessian_fd_problem = None
        problem = self._auxiliary_mdo_formulation.optimization_problem
        if self.__second_order:
            self.__hessian_fd_problem = OptimizationProblem(self.uncertain_space)
            self.__hessian_fd_problem.objective = HessianFunction(problem.objective)

        problem.differentiation_method = differentiation_method
        problem.design_space = problem.design_space.to_design_space()
        self.optimization_problem.differentiation_method = (
            self.optimization_problem.ApproximationMode.FINITE_DIFFERENCES
        )
        self.optimization_problem.fd_step = 1e-6

    @property
    def hessian_fd_problem(self) -> OptimizationProblem | None:
        """The problem related to the approximation of the Hessian."""
        return self.__hessian_fd_problem

    @property
    def second_order(self) -> bool:
        """Whether to use a second order approximation."""
        return self.__second_order

    def add_constraint(  # noqa: D102
        self,
        output_name: str | Sequence[str],
        statistic_name: str,
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.INEQ,
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

    def evaluate_with_mean(self, problem: OptimizationProblem, eval_jac: bool) -> None:
        """Evaluate the functions at the mean value of the uncertain vector.

        Args:
            problem: The problem including the functions.
            eval_jac: Whether to evaluate the Jacobian functions.
        """
        objective = problem.objective
        if objective is objective.original:
            problem.preprocess_functions(
                is_function_input_normalized=False, eval_obs_jac=eval_jac
            )
        output_functions, jacobian_functions = problem.get_functions(
            observable_names=(), jacobian_names=() if eval_jac else None
        )
        problem.evaluate_functions(
            self._uncertain_space.distribution.mean,
            output_functions=output_functions,
            jacobian_functions=jacobian_functions,
        )
