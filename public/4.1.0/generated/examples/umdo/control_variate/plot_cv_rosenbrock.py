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
r"""# The Rosenbrock mono-disciplinary problem

In this example, we consider the Rosenbrock mono-disciplinary optimization problem

$$\min_{x,y\in[-2,2]} \mathbb{E}[(U-x)^2+100(y-x^2)^2]$$

where $U\sim\mathcal{N}(0,0.0025)$ is a Gaussian variable and $\mathbb{E}$ is the
expectation operator.

In the following, we will call $f$ the function computing $(U-x)^2+100(y-x^2)^2$ given
$x$, $y$ and $U$.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline

from gemseo_umdo.formulations.control_variate_settings import ControlVariate_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()

# %%
# Firstly,
# we define the discipline implementing the Rosenbrock function $f$:
discipline = AnalyticDiscipline({"z": "(u-x)**2+100*(y-x**2)**2"}, name="Rosenbrock")

# %%
# where $x,y$ belongs to the interval $[-2,2]$:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2, upper_bound=2.0, value=-2.0)
design_space.add_variable("y", lower_bound=-2, upper_bound=2.0, value=-2.0)

# %%
# and $U$ is a Gaussian variable with unit mean
# and standard deviation equal to 0.05:
uncertain_space = ParameterSpace()
uncertain_space.add_random_variable("u", "OTNormalDistribution", mu=1.0, sigma=0.05)

# %%
# Then,
# we define a [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
# to minimize the statistic $\mathbb{E}[(U-x)^2+100(y-x^2)^2]$
# estimated using a control variates technique
# based on Taylor polynomials and 50 samples at each iteration of the optimization loop:
scenario = UMDOScenario(
    [discipline],
    "z",
    design_space,
    uncertain_space,
    "Mean",
    formulation_name="DisciplinaryOpt",
    statistic_estimation_settings=ControlVariate_Settings(n_samples=30),
)

# %%
# We execute it with the gradient-based optimizer SLSQP:
#
# !!! warning
#     The implementation of statistic estimators do not allow for the moment
#     to use analytical derivatives.
#     Please use finite differences or complex step to approximate the gradients.
scenario.set_differentiation_method("finite_differences")
scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)

# %%
# and plot the optimization history:
scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Lastly,
# we can compare the numerical solution of this Rosenbrock problem under uncertainty
(scenario.optimization_result.x_opt, scenario.optimization_result.f_opt)

# %%
# to the solution of the Rosenbrock problem without uncertainty,
# namely
# $(x^*,f^*)=([1, 1], 0)$.
