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
r"""Taylor polynomials for multidisciplinary design problems under uncertainty.

:class:`.TaylorPolynomial` is an
:class:`~gemseo_umdo.formulations.formulation.UMDOFormulation`
estimating the statistics with first- or second-order Taylor polynomials
around the expectation of the uncertain variables:
:math:`f(x,U)\approx f(x,\mu) + (U-\mu)f'(x,\mu) \pm 0.5(U-\mu)^2f''(x,\mu)`.

E.g.
:math:`\mathbb{E}[f(x,U)]\approx
\frac{1}{N}\sum_{i=1}^N f\left(x,U^{(i)}\right)`
or
:math:`\mathbb{V}[f(x,U)]\approx \sigma^2f'(x,\mu)`
where :math:`U` is normally distributed
with mean :math:`\mu` and unit variance :math:`\sigma`.
"""
from __future__ import annotations

import logging
from typing import Any
from typing import ClassVar
from typing import Mapping
from typing import Sequence

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.lib_custom import CustomDOE
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.formulation import MDOFormulation
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.derivatives.finite_differences import FirstOrderFD
from gemseo.utils.logging_tools import LoggingContext
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import ndarray

from gemseo_umdo.estimators.taylor_polynomial import (
    TaylorPolynomialEstimatorFactory,
)
from gemseo_umdo.formulations.formulation import UMDOFormulation

LOGGER = logging.getLogger(__name__)


class TaylorPolynomial(UMDOFormulation):
    """Robust MDO formulation based on Taylor polynomials."""

    _STATISTIC_FACTORY: ClassVar = TaylorPolynomialEstimatorFactory()

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: MDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        objective_statistic_parameters: Mapping[str, Any] | None = None,
        maximize_objective: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        differentiation_method: OptimizationProblem.DifferentiationMethod = OptimizationProblem.DifferentiationMethod.USER_GRAD,  # noqa: B950
        second_order: bool = False,
        **options: Any,
    ) -> None:
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
            grammar_type=grammar_type,
            **options,
        )

        self.__hessian_fd_problem = None
        finite_differences = self.opt_problem.ApproximationMode.FINITE_DIFFERENCES
        if self.__second_order:
            problem = self._mdo_formulation.opt_problem
            self.__hessian_fd_problem = OptimizationProblem(self.uncertain_space)
            self.__hessian_fd_problem.objective = HessianFunction(problem.objective)

        problem = self._mdo_formulation.opt_problem
        problem.differentiation_method = differentiation_method
        problem.design_space = problem.design_space.to_design_space()
        self.opt_problem.differentiation_method = finite_differences
        self.opt_problem.fd_step = 1e-6
        self.__custom_doe = CustomDOE()

    @property
    def hessian_fd_problem(self) -> OptimizationProblem:
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
        constraint_type: str = MDOFunction.ConstraintType.INEQ,
        constraint_name: str | None = None,
        value: float | None = None,
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
            self.hessian_fd_problem.add_constraint(
                HessianFunction(self.mdo_formulation.opt_problem.constraints[-1]),
                cstr_type=MDOFunction.ConstraintType.INEQ,
            )

    def add_observable(  # noqa: D102
        self,
        output_names: Sequence[str],
        statistic_name: str,
        observable_name: Sequence[str] | None = None,
        discipline: MDODiscipline | None = None,
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
                HessianFunction(self.mdo_formulation.opt_problem.observables[-1]),
            )

    def evaluate_with_mean(self, problem: OptimizationProblem, eval_jac: bool) -> None:
        """Evaluate the functions of a problem at the mean of the uncertain variables.

        Args:
            problem: The problem.
            eval_jac: Whether to evaluate the Jacobian functions.
        """
        with LoggingContext():
            self.__custom_doe.execute(
                problem,
                samples=self._uncertain_space.distribution.mean[None],
                eval_jac=eval_jac,
                eval_obs_jac=eval_jac,
            )

    class _StatisticFunction(UMDOFormulation._StatisticFunction):
        def _func(self, input_data: ndarray) -> ndarray:
            formulation = self._formulation
            problem = formulation.mdo_formulation.opt_problem
            if self._function_name in formulation._processed_functions:
                formulation._processed_functions = []
                problem.reset()
                if formulation.hessian_fd_problem is not None:
                    formulation.hessian_fd_problem.reset()

            database = problem.database
            if not database:
                formulation.update_top_level_disciplines(input_data)
                formulation.evaluate_with_mean(problem, True)
                if formulation.hessian_fd_problem is not None:
                    formulation.evaluate_with_mean(
                        formulation.hessian_fd_problem, False
                    )

            func_value = atleast_1d(
                database.get_function_history(self._function_name)[0]
            )
            jac_value = atleast_2d(
                database.get_gradient_history(self._function_name)[0]
            )
            hess_value = None
            if formulation.second_order:
                hessian_database = formulation.hessian_fd_problem.database
                hess_name = (
                    f"{database.GRAD_TAG}{database.GRAD_TAG}{self._function_name}"
                )
                hess_value = hessian_database.get_function_history(hess_name)[0]
                if hess_value.ndim == 1:
                    hess_value = hess_value[None, None, ...]

                if hess_value.ndim == 2:
                    hess_value = hess_value[None, ...]

            formulation._processed_functions.append(self._function_name)
            return self._estimate_statistic(
                func_value,
                jac_value,
                hess_value,
                **self._statistic_parameters,
            )


class HessianFunction(MDOFunction):
    """Approximation of the Hessian function with finite differences.

    Take an original function and approximate its Hessian with finite differences
    applied to its analytical or approximated Jacobian.
    """

    def __init__(self, func: MDOFunction) -> None:
        """
        Args:
            func: The original function.
        """  # noqa: D205 D212 D415
        self.__jac = func.jac if func.has_jac else FirstOrderFD(func.func).f_gradient
        grad_tag = Database.GRAD_TAG
        super().__init__(
            FirstOrderFD(self._compute_jac).f_gradient,
            f"{grad_tag}{grad_tag}{func.name}",
        )

    def _compute_jac(self, input_data: ndarray) -> ndarray:
        """Compute the Jacobian matrix.

        Args:
            input_data: The input data.

        Returns:
            The Jacobian matrix.
        """
        return self.__jac(input_data).T
