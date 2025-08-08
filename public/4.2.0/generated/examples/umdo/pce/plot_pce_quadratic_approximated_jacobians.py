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
r"""# A quadratic mono-disciplinary problem with approximated statistics Jacobian

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
from gemseo.disciplines.auto_py import AutoPyDiscipline

from gemseo_umdo.formulations.pce_settings import PCE_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()


# %%
# Firstly,
# we define an [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline]
# implementing the function $f$:
def f(x, u):
    y = (x + u) ** 2
    return y


discipline = AutoPyDiscipline(f)


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
# estimated using a polynomial chaos expansion (PCE)
# trained from 20 samples at each iteration of the optimization loop:
scenario = UMDOScenario(
    [discipline],
    "y",
    design_space,
    uncertain_space,
    "Mean",
    formulation_name="DisciplinaryOpt",
    statistic_estimation_settings=PCE_Settings(
        n_samples=20, approximate_statistics_jacobians=True
    ),
)

# %%
# !!! note
#     The mean, standard deviation and variance
#     of the PCE built over the uncertain space
#     are differentiable with respect to the design variables
#     if both the disciplines and the MDO formulation are differentiable,
#     which is the case in this example.
#     This implies that a gradient-based optimization algorithm can be used
#     without approximating the derivatives of the objective.
#     If this is not the case
#     or if we do not want to compute the derivatives of the disciplines
#     to reduce the calculation budget,
#     we can activate the `approximate_statistics_jacobians` option as done above
#     to use the approximation technique
#     proposed in Section II.C.3 of a paper by Mura _et al._[@Mura2020]
#     and parametrized by the `differentiation_step` option (default: `1e-6`) .
#
# We execute this scenario using the gradient-base optimizer SLSQP:
scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)

# %%
# and plot the optimization history:
scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Notice that the numerical solution
(scenario.optimization_result.x_opt[0], scenario.optimization_result.f_opt)

# %%
# is close to the theoretical solution $(x^*,\mathbb{E}[(x^*+U)^2])=(0,1)$.
