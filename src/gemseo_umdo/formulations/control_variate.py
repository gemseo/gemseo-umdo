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

[ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate] is an
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
from typing import Any

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.logging_tools import LoggingContext
from gemseo.utils.seeder import SEED

from gemseo_umdo.formulations._functions.statistic_function_for_control_variate import (
    StatisticFunctionForControlVariate,
)
from gemseo_umdo.formulations._statistics.control_variate.factory import (  # noqa: E501
    ControlVariateEstimatorFactory,
)
from gemseo_umdo.formulations.base_umdo_formulation import BaseUMDOFormulation

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.doe.doe_library import DOELibrary
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.formulations.mdo_formulation import MDOFormulation
    from gemseo.typing import RealArray


class ControlVariate(BaseUMDOFormulation):
    """Control variate-based U-MDO formulation.

    !!! note "DOE algorithms"
        This formulation uses a DOE algorithm;
        read the
        [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/algorithms/doe_algos.html).
        for more information about the available DOE algorithm names and options.
    """

    __doe_algo: DOELibrary
    """The DOE algorithm to sample the uncertain problem."""

    __doe_algo_options: dict[str, Any]
    """The options of the DOE algorithm."""

    __linearization_problem: OptimizationProblem
    """The problem related to the linearization of the functions used as control
    variates."""

    __n_samples: int
    """The number of samples."""

    __seed: int
    """The seed for reproducibility."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        mdo_formulation: MDOFormulation,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        n_samples: int,
        objective_statistic_parameters: Mapping[str, Any] = READ_ONLY_EMPTY_DICT,
        maximize_objective: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        algo: str = OpenTURNS.OT_LHSO,
        algo_options: Mapping[str, Any] = READ_ONLY_EMPTY_DICT,
        seed: int = SEED,
        **options: Any,
    ) -> None:
        """
        Args:
            n_samples: The number of samples, i.e. the size of the DOE.
            algo: The name of the DOE algorithm.
            algo_options: The options of the DOE algorithm.
            seed: The seed for reproducibility.
        """  # noqa: D205 D212 D415
        self._statistic_function_class = StatisticFunctionForControlVariate
        self._statistic_factory = ControlVariateEstimatorFactory()
        self.__doe_algo = DOELibraryFactory().create(algo)
        self.__doe_algo_options = dict(algo_options)
        self.__doe_algo_options["n_samples"] = n_samples
        self.__n_samples = n_samples
        self.__seed = seed
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
        mdo_formulation = self._mdo_formulation
        self.__linearization_problem = OptimizationProblem(self.uncertain_space)
        self.__linearization_problem.objective = (
            mdo_formulation.optimization_problem.objective
        )
        self.name = (
            f"{self.__class__.__name__}"
            f"[{mdo_formulation.__class__.__name__}; {algo}({n_samples})]"
        )

    @property
    def linearization_problem(self) -> OptimizationProblem:
        """The problem related to the linearization of the functions."""
        return self.__linearization_problem

    @property
    def doe_algo(self) -> DOELibrary:
        """The DOE library configured with an algorithm."""
        return self.__doe_algo

    def _post_add_constraint(self) -> None:
        self.__linearization_problem.add_observable(
            self._mdo_formulation.optimization_problem.observables[-1]
        )

    def _post_add_observable(self) -> None:
        self.__linearization_problem.add_observable(
            self._mdo_formulation.optimization_problem.observables[-1]
        )

    def compute_samples(
        self, problem: OptimizationProblem, input_data: RealArray
    ) -> None:
        """Evaluate the functions of a problem with a DOE algorithm.

        Args:
            problem: The problem.
            input_data: The input point at which to estimate the statistic.
        """
        with LoggingContext(logging.getLogger("gemseo")):
            self.__doe_algo.seed = self.__seed
            self.__doe_algo.execute(problem, **self.__doe_algo_options)

    def evaluate_with_mean(self) -> None:
        """Evaluate the Taylor polynomials at the mean value of the uncertain vector."""
        if self.__linearization_problem.nonproc_objective is None:
            self.__linearization_problem.preprocess_functions(
                is_function_input_normalized=False, eval_obs_jac=True
            )
        self.__linearization_problem.evaluate_functions(
            self._uncertain_space.distribution.mean, eval_jac=True
        )
