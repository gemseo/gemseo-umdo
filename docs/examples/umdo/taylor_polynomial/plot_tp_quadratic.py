# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
r"""# A quadratic mono-disciplinary problem

In this example, we consider the quadratic mono-disciplinary optimization problem

$$\min_{x\in[-1,1]} \mathbb{E}[(x+U)^2]$$

where $U\sim\mathcal{N}(0,1)$ is a standard Gaussian variable and $\mathbb{E}$ is the
expectation operator.

The objective can be rewritten as $x^2+1$ and then the solution is obvious, namely

$$(x^*,\mathbb{E}[(x^*+U)^2])=(0,1).$$

In the following, we will call $f$ the function computing $(x+U)^2$ given $x$ and $U$.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline

from gemseo_umdo.formulations.taylor_polynomial_settings import (
    TaylorPolynomial_Settings,
)
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()

# %%
# Firstly,
# we define an [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline]
# implementing the function $f$:
discipline = AnalyticDiscipline({"y": "(x+u)**2"}, name="f")

# %%
# as well as the design space:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)

# %%
# and the uncertain space:
uncertain_space = ParameterSpace()
uncertain_space.add_random_variable("u", "OTNormalDistribution")

# %%
# Then,
# we define a [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
# to minimize the statistic $\mathbb{E}[(x+U)^2]$
# estimated using a first-order Taylor polynomial of $f$ at $\mathbb{E}[U]=0$
# at each iteration of the optimization loop:
scenario = UMDOScenario(
    [discipline],
    "y",
    design_space,
    uncertain_space,
    "Mean",
    formulation_name="DisciplinaryOpt",
    statistic_estimation_settings=TaylorPolynomial_Settings(),
)

# %%
# We execute this scenario using the gradient-free optimizer COBYLA:
scenario.execute(algo_name="NLOPT_COBYLA", max_iter=100)

# %%
# and plot the optimization history:
scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Notice that the numerical solution
(scenario.optimization_result.x_opt[0], scenario.optimization_result.f_opt)

# %%
# is far from the theoretical solution $(x^*,\mathbb{E}[(x^*+U)^2])=(0,1)$
# in terms of objective value,
# as the objective function is far from being quadratic with respect to $U$.
# However,
# we can see that the optimum design value is equal to the theoretical one.
#
# On the other hand,
# we can use a second-order Taylor polynomial
scenario = UMDOScenario(
    [discipline],
    "y",
    design_space,
    uncertain_space,
    "Mean",
    formulation_name="DisciplinaryOpt",
    statistic_estimation_settings=TaylorPolynomial_Settings(second_order=True),
)
scenario.execute(algo_name="NLOPT_COBYLA", max_iter=100)

# %%
# and see that it gives the exact solution:
(scenario.optimization_result.x_opt[0], scenario.optimization_result.f_opt)

# %%
# which is quite logical in this simple example
# since the function $f$ is quadratic with respect to $U$.
