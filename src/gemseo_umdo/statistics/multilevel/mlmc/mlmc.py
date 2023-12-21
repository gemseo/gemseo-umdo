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
"""A generic algorithm for multilevel Monte Carlo (MLMC) sampling."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.timer import Timer
from matplotlib import pyplot as plt
from numpy import array
from numpy import cumsum
from numpy import isnan
from numpy import nan
from numpy import zeros

from gemseo_umdo.monte_carlo_sampler import MonteCarloSampler
from gemseo_umdo.statistics.multilevel.mlmc.pilots.factory import MLMCPilotFactory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.base_factory import BaseFactory
    from numpy.typing import NDArray

    from gemseo_umdo.statistics.multilevel.mlmc.level import Level
    from gemseo_umdo.statistics.multilevel.pilot import Pilot

LOGGER = logging.getLogger(__name__)


class MLMC:
    r"""Multilevel Monte Carlo (MLMC) algorithm.

    This algorithm aims at sampling the different model levels in an adaptive way,
    with many evaluations for the coarsest model
    and a few evaluations for the finest one.

    This adaptive sampling is guided
    by a [Pilot][gemseo_umdo.statistics.multilevel.pilot.Pilot].

    This algorithm depends on the execution cost ratio between two consecutive levels,
    that can be estimated from the models,
    and on the sampling size ratio between two sampling steps on the same level.

    At a given iteration,
    the algorithm

    1. considers a level $\ell^*$ and a sample size $n_{\ell^*}$
    2. samples the models involved in the statistic $T_{\ell^*}$
       of the telescoping sum (TS) $\theta_L = \sum_{\ell=0}^L T_\ell$,
    3. computes the new level $\ell^*$ to sample
       and the corresponding sample size $n_{\ell^*}$.
    """

    _algorithm_name: str
    """The name of the algorithm."""

    _n_levels: int
    r"""The number of levels $L+1$ of the telescopic sum."""

    _PILOT_FACTORY: ClassVar[type[BaseFactory]] = MLMCPilotFactory
    """The factory of pilot statistics."""

    _samplers: tuple[MonteCarloSampler]
    """The Monte Carlo samplers of each level of the telescopic sum."""

    __budget_history: list
    """The history of the budget."""

    __C_l: NDArray[float | nan]
    r"""The evaluations cost of the models."""

    __costs: NDArray[float]
    r"""The unit sampling costs of each level of the telescopic sum.

    Namely,
    $(\mathcal{C}_{\ell-1}+\mathcal{C}_\ell)_{\ell\in\{0,\ldots,L\}}$
    with $\mathcal{C}_{-1}=0$.
    """

    __current_budget: float
    """The current budget."""

    __delta_n_l: list[int]
    """The current additional numbers of samples of each level."""

    __f_l: list[MDOFunction]
    r"""The models $f_0,f_1,\ldots,f_L$."""

    __minimum_budget: float
    """The minimum cost of the algorithm given the initial sample sizes per level."""

    __n_l: NDArray[int]
    r"""The total number of samples per level, from $\ell=0$ to $\ell=L$."""

    __n_samples_history: list[list[int]]
    """The history of the additional numbers of samples of each level."""

    __pilot_statistic_estimation: NDArray[float]
    """The estimation of the pilot statistic."""

    __pilot_statistic_estimator: Pilot
    """The estimator of the pilot statistic."""

    _pilot_statistic_estimator_parameters: list[Any]
    """The parameters of the estimator of the pilot statistic."""

    __r_l: NDArray[float]
    r"""The sampling ratios of each level of the telescopic sum.

    Namely, $r_0,r_1,\ldots,r_L$.
    """

    __seed: int
    """The initial seed used by the Monte Carlo samplers.

    Then, the seed is incremented at each level of the telescopic sum and at each
    algorithm iteration.
    """

    __total_budget: float
    """The maximum cost given by the user."""

    __total_execution_times: NDArray[float]
    """The total execution times of the different models."""

    __use_empirical_C_l: bool  # noqa: N815
    r"""Whether the allocation criterion uses empirical $C_\ell$."""

    def __init__(
        self,
        levels: Iterable[Level],
        uncertain_space: ParameterSpace,
        n_samples: float,
        pilot_statistic_name: str = "Mean",
        seed: int = 0,
    ) -> None:
        r"""
        Args:
            levels: The levels
                defined in terms of model, evaluation cost and initial number of calls.
            uncertain_space: The uncertain space on which to sample the functions.
            n_samples: The sampling budget expressed as
                the number of model evaluations
                equivalent to evaluations of the finest model.
                This number is not necessarily an integer;
                for instance,
                if $f_L$ is twice as expensive as $f_{L-1}$,
                then ``n_samples=1.5`` can correspond to
                1 evaluation of $f_L$ and 1 evaluation of $f_{L-1}$.
            pilot_statistic_name: The name of the statistic used to drive the algorithm.
            seed: The initial random seed for reproducibility.
                Then,
                the seed is incremented at each level of the telescopic sum
                and at each algorithm iteration.

        Raises:
            ValueError: When the minimum cost is greater than the maximum cost.
        """  # noqa: D205 D212 D415
        self._algorithm_name = "MLMC"

        # Initialize the seed.
        self.__seed = seed

        # Set the models f_0, f_1, ..., f_L.
        self.__f_l = [level.model for level in levels]
        for l, f_l in enumerate(self.__f_l):  # noqa: E741
            f_l.name = f"f[{l}]"

        # Set the number of levels.
        self._n_levels = len(self.__f_l)

        # Set the unit sampling costs of each level of the telescopic sum (TS).
        C_l = array(  # noqa: N806
            [level.cost if level.cost is not None else nan for level in levels]
        )
        self.__C_l = C_l = C_l / C_l[-1]  # noqa: N806
        self.__total_execution_times = array([0] * self._n_levels)
        self.__costs = array([C_l[0], *(C_l[1:] + C_l[:-1]).tolist()])

        # Set the sampling ratios r_l of each level of the TS.
        self.__r_l = array([level.sampling_ratio for level in levels])

        # Set the Monte Carlo samplers of each level of the TS.
        self._samplers = tuple(
            MonteCarloSampler(uncertain_space) for _ in range(self._n_levels)
        )
        self._add_functions_to_samplers()

        # Set the numbers of samples to be added at each level of the TS.
        self.__delta_n_l = [level.n_initial_samples for level in levels]

        # Initialize the history of numbers of samples added at each level of the TS.
        self.__n_samples_history = [self.__delta_n_l.copy()]

        # Initialize the numbers of samples of each level of the TS.
        self.__n_l = array(self.__n_samples_history[0], dtype="int64")

        self.__minimum_budget = sum(
            nl * cost for nl, cost in zip(self.__n_samples_history[0], self.__costs)
        )
        self.__total_budget = n_samples
        self.__current_budget = self.__total_budget
        self.__budget_history = []
        self.__use_empirical_C_l = isnan(self.__minimum_budget)
        if not self.__use_empirical_C_l and self.__minimum_budget > n_samples:
            raise ValueError(
                f"The minimum budget {self.__minimum_budget} is greater "
                f"than the total budget {n_samples}."
            )

        # Set the estimator of the pilot statistic and initialize its estimation.
        self.__pilot_statistic_estimation = array([])
        self.__pilot_statistic_estimator = self._PILOT_FACTORY().create(
            pilot_statistic_name,
            sampling_ratios=self.__r_l,
            costs=self.__costs,
        )
        self._pilot_statistic_estimator_parameters = []
        self.__V_l = 0
        LOGGER.info("%s", self)

    def __repr__(self) -> str:
        string = MultiLineString()
        string.add("Algorithm {}", self._algorithm_name)
        string.indent()
        string.add("Number of levels: {}", self._n_levels)
        string.add(
            "Pilot statistic: {}", self.__pilot_statistic_estimator.__class__.__name__
        )
        string.add("Budget")
        string.indent()
        string.add("Minimum: {}", self.__minimum_budget)
        string.add("Maximum: {}", self.__total_budget)
        string.dedent()
        string.add("Numbers of initial samples")
        string.indent()
        for level, n_samples in enumerate(self.__n_samples_history[0]):
            string.add("n_{} = {}", level, n_samples)
        string.dedent()
        string.add("Evaluation costs of the models")
        string.indent()
        for level, cost in enumerate(self.__C_l):
            string.add("C_{} = {}", level, cost)
        string.dedent()
        string.add("Evaluation costs of the levels")
        string.indent()
        for level, cost in enumerate(self.__costs):
            if level == 0:
                string.add("C_{} = {}", level, cost)
            else:
                string.add("C_{} + C_{} = {}", level, level - 1, cost)
        string.dedent()
        string.add("Sampling ratios:")
        string.indent()
        for level, r_l in enumerate(self.__r_l):
            string.add("r_{} = {}", level, r_l)
        string.dedent()
        return str(string)

    @property
    def pilot_statistic_estimation(self) -> NDArray[float]:
        """The estimation of the pilot statistic."""
        return self.__pilot_statistic_estimation

    @property
    def sampling_history(self) -> NDArray[int]:
        """The history of the numbers of samples of each level of the telescopic sum.

        ``algo.sampling_size_history[i, l]`` is the number of samples
        at iteration ``i+1`` and level ``l``.
        """
        return array(self.__n_samples_history)

    @property
    def budget_history(self) -> NDArray[float]:
        """The history of the budget.

        ``algo.budget_history[i]`` is the budget at iteration ``i+1``.
        """
        return array(self.__budget_history)

    @property
    def n_total_samples(self) -> NDArray[int]:
        """The total numbers of samples per level.

        ``algo.n_total_samples[l]`` is the total number of samples at level ``l``.
        """
        return self.__n_l

    @property
    def model_costs(self) -> NDArray[float]:
        """The evaluation costs of the different models.

        ``algo.model_costs[l]`` is the cost of one evaluation of the ``l``-th model.
        """
        return self.__C_l

    @property
    def level_costs(self) -> NDArray[float]:
        """The evaluation costs of the different levels.

        ``algo.level_costs[l]`` is the cost of one evaluation of the ``l``-th level.
        """
        return self.__costs

    def _add_functions_to_samplers(self) -> None:
        """Add the functions to the samplers."""
        # At level l>0, sample the functions f_l and f_{l-1}.
        for f_l, f_l_1, sampler in zip(
            self.__f_l[1:], self.__f_l[:-1], self._samplers[1:]
        ):
            sampler.add_function(f_l)
            sampler.add_function(f_l_1)

        # At level 0, sample the functions f_0 and f_{-1}: x -> 0.
        self._samplers[0].add_function(self.__f_l[0])
        self._samplers[0].add_function(MDOFunction(self.__zero_function, "f[-1]"))

    @staticmethod
    def __zero_function(x: NDArray[float]) -> NDArray[float]:
        """A function returning 0 for any input value.

        Args:
            x: The input value,
                shaped as ``(n_features,)`` or ``(n_samples, n_features)``.

        Returns:
            The output value,
            shaped as ``(1,)`` or ``(n_samples, 1)``.
        """
        return array([0.0] if x.ndim == 1 else [[0.0]] * len(x))

    def execute(self) -> None:
        """Execute the algorithm."""
        # The current version of the algorithm samples only one level at a time,
        # except at the first iteration where it samples them all.
        levels_to_be_sampled = list(range(self._n_levels))

        # Initialize the iteration of the algorithm.
        is_last_iteration = False
        iteration = 0

        # As long as there is budget left
        LOGGER.info("Start sampling with a total budget of %s", self.__total_budget)
        while self.__current_budget >= 0:
            iteration += 1
            if is_last_iteration:
                LOGGER.info("   Iteration #%s (last iteration)", iteration)
            else:
                LOGGER.info("   Iteration #%s", iteration)

            # Append the budget to the budget history.
            self.__budget_history.append(self.__current_budget)

            # Sample the selected levels of the TS.
            levels_to_samples = self.__compute_samples(*levels_to_be_sampled)

            # Select the next level l_star of the TS to be sampled
            # and estimate the statistic.
            (
                l_star,
                self.__pilot_statistic_estimation,
            ) = self.__pilot_statistic_estimator.compute_next_level_and_statistic(
                levels_to_be_sampled,
                self.__n_l,
                levels_to_samples,
                *self._pilot_statistic_estimator_parameters,
            )

            # Stop the algorithm if it is the last iteration.
            if is_last_iteration:
                break

            # The current version of the algorithm samples only one level at a time.
            levels_to_be_sampled = [l_star]

            # Define the corresponding sample size.
            delta_n_l_star = int(
                math.floor((self.__r_l[l_star] - 1) * self.__n_l[l_star])
            )
            n_l_star = self.__n_l[l_star] + delta_n_l_star
            LOGGER.info("      Find the next level to sample")
            LOGGER.info("         l_star = %s", l_star)
            LOGGER.info("         d_n_l_star = %s", delta_n_l_star)
            LOGGER.info("         n_l_star = %s", n_l_star)

            # If the new sampling stage is too expensive, reduce the number of samples.
            posterior_budget = (
                self.__current_budget - delta_n_l_star * self.__costs[l_star]
            )
            if posterior_budget < 0:
                LOGGER.info("         Maximum budget exceeded by %s", -posterior_budget)
                LOGGER.info(
                    "         Decrease d_n_l_star to respect the maximum budget"
                )

                # There is a budget to do at most one iteration.
                is_last_iteration = True

                # Update the numbers of additional samples at level l_star
                # to achieve a positive or zero budget.
                delta_n_l_star = int(
                    delta_n_l_star + posterior_budget / self.__costs[l_star]
                )
                n_l_star = self.__n_l[l_star] + delta_n_l_star
                LOGGER.info("         d_n_l_star = %s", delta_n_l_star)
                LOGGER.info("         n_l_star = %s", n_l_star)
                # Stop the algorithm if one can no longer sample l_star.
                if delta_n_l_star == 0:
                    LOGGER.info(
                        "Stop the algorithm as sampling l_star is too expensive."
                    )
                    break

            # Update the history of number of samples of each level
            # (0 for all the levels, but l_star).
            self.__delta_n_l = zeros(self._n_levels)
            self.__delta_n_l[l_star] = delta_n_l_star
            self.__n_l[l_star] += delta_n_l_star
            self.__n_samples_history.extend([self.__delta_n_l.copy()])

        LOGGER.info("Sampling completed")
        LOGGER.info("Results")
        LOGGER.info("   Pilot statistic = %s", self.pilot_statistic_estimation)
        LOGGER.info("   Total cost = %s", sum(self.__n_l * self.__costs))
        LOGGER.info("   Cost allocation")
        levels_to_total_costs = self.__n_l * self.__costs
        levels_to_total_costs = levels_to_total_costs / sum(levels_to_total_costs)
        for level, total_cost in enumerate(levels_to_total_costs):
            LOGGER.info("      Level %s: %s", level, f"{total_cost:.1%}")

        LOGGER.info("   n_l")
        for level in range(self._n_levels):
            LOGGER.info("       n_%s = %s", level, self.__n_l[level])

        LOGGER.info("   V_l")
        self.__V_l = self.__pilot_statistic_estimator.V_l
        for level in range(self._n_levels):
            LOGGER.info("       V_%s = %s", level, f"{self.__V_l[level]:.2e}")

    def __compute_samples(self, *levels_to_be_sampled: int) -> list[NDArray[float]]:
        """Sample the low- & high-fidelity models at some levels of the telescoping sum.

        Args:
            levels_to_be_sampled: The levels of the telescoping sum to be sampled.

        Returns:
            The model samples.
        """
        delta_n_l_star = [int(x) for x in self.__delta_n_l]
        LOGGER.info("      Sampling")
        for level in range(self._n_levels):
            LOGGER.info("         delta_n_%s = %s", level, delta_n_l_star[level])
        levels_to_samples = [array([]) for _ in range(self._n_levels)]
        for level in levels_to_be_sampled:
            self.__seed += 1
            sampler = self._samplers[level]
            with Timer() as timer:
                sampler(delta_n_l_star[level], self.__seed)

            self.__total_execution_times[level] += timer.elapsed_time
            levels_to_samples[level] = sampler.output_history

        if self.__use_empirical_C_l:
            self.__C_l = self.__total_execution_times / self.__total_execution_times[-1]
            self.__costs = array([
                self.__C_l[0],
                *(self.__C_l[1:] + self.__C_l[:-1]).tolist(),
            ])
        cost = sum(delta_n_l_star * self.__costs)
        LOGGER.info("         Cost = %s", cost)
        self.__current_budget -= cost
        LOGGER.info("         Remaining budget = %s", self.__current_budget)
        return levels_to_samples

    def plot_evaluation_history(
        self,
        show: bool = True,
        file_path: str | Path | None = None,
        log_n_evaluations: bool = True,
        log_budget: bool = False,
    ) -> None:
        """Plot the history of the model evaluations in terms of sample size and budget.

        Args:
            show: Whether to display the graph.
            file_path: The file path to save the graph.
            log_n_evaluations: Whether to use a log-scale for the number of evaluations.
            log_budget: Whether to use a log-scale for the budget.
        """
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        iterations = [i + 1 for i, _ in enumerate(self.__n_samples_history)]
        ax1.plot(
            iterations,
            cumsum(array(self.__n_samples_history), axis=0),
            label=[rf"$f_{level}$" for level in range(self._n_levels)],
            marker=".",
        )
        if log_n_evaluations:
            ax1.set_yscale("log")

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cumulated number of evaluations")
        ax1.legend(title="Simulators")
        ax1.grid(which="both")
        data = (array(self.__n_samples_history) * self.__costs).T
        ax2.bar(iterations, data[0], label=r"$f_0$")
        for index, row in enumerate(data[1:]):
            ax2.bar(
                iterations,
                row,
                bottom=data[0 : index + 1].sum(0),
                label=rf"$f_{index + 1}$",
            )

        if log_budget:
            ax2.set_yscale("log")

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Cost")
        ax2.legend(title="Simulators")
        ax2.grid(which="both")
        ax2.set_axisbelow(True)
        save_show_figure(fig, show, file_path, fig_size=(10, 3))
