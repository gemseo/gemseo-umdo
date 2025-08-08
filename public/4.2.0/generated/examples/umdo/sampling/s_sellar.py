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
r"""# The Sellar's MDO problem

In this example,
we consider the Sellar's MDO problem under uncertainty

$$\min_{x,z_1,x_2} \mathbb{E}[f(x,z_2,y_1,y_2)]$$

over the design space $[0,10]\times[-10,10]\times[0,10]$ and
under the inequality constraints

$$\mathbb{E}[c_1(y_1)]+3\mathbb{S}[c_1(y_1)] \leq 0$$

and

$$\mathbb{E}[c_2(y_2)]+3\mathbb{S}[c_2(y_2)] \leq 0,$$

where

- $\mathbb{E}$ is the expectation operator,
- $\mathbb{S}$ is the standard deviation operators,
- $f(x,z_2) = x^2 + z_2 + y_1^2 + \exp(-y_2)$ is the objective function,
- $c_1(y_1) = 3.16 - y_1^2$ is the first constraint function,
- $c_2(y_2) = y_2 - 24.0$ is the second constraint function,
- $y_1 = \sqrt{z_1^2 + z_2 + x - ay_2}$ is the first coupling equation,
- $y_2 = \frac{\log(1+\exp(10y_1))}{5} - y_1 - \frac{\log(2)}{5} + z_1 + z_2$
  is the second coupling equation,
- $a$ is a random variable distributed
  according to the triangular distribution $\mathcal{T}(0.1,0.2,0.3)$.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()

# %%
# Firstly,
# we create one discipline per coupling equation
# and a system discipline to compute the objective and constraints:
system = AnalyticDiscipline({
    "obj": "x**2 + z2 + y1**2 + exp(-y2)",
    "c1": "3.16 - y1 ** 2",
    "c2": "y2 - 24.0",
})
disc1 = AnalyticDiscipline({"y1": "(z1**2 + z2 + x - a*y2)**0.5"})
disc2 = AnalyticDiscipline({"y2": "abs(y1) + z1 + z2"})

# %%
# as well as the design space:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
design_space.add_variable("z1", lower_bound=-10.0, upper_bound=10.0, value=4.0)
design_space.add_variable("z2", lower_bound=0.0, upper_bound=10.0, value=3.0)

# %%
# Secondly,
# we define the uncertain space:
uncertain_space = ParameterSpace()
# %%
# with an uncertainty over the constant `"a"`:
uncertain_space.add_random_variable(
    "a", "OTTriangularDistribution", minimum=0.1, maximum=0.3, mode=0.2
)

# %%
# Then,
# we define a [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
# to minimize the statistic $\mathbb{E}[f(x,z_2,y_1,y_2)]$
# estimated using a crude Monte Carlo sampling strategy
# with 100 samples at each iteration of the optimization loop:
scenario = UMDOScenario(
    [system, disc1, disc2],
    "obj",
    design_space,
    uncertain_space,
    "Mean",
    formulation_name="MDF",
    statistic_estimation_settings=Sampling_Settings(n_samples=100),
)

# %%
# while satisfying the constraints
# $\mathbb{E}[c_1(y_1)]+3\mathbb{S}[c_1(y_1)] \leq 0$
# and
# $\mathbb{E}[c_2(y_2)]+3\mathbb{S}[c_2(y_2)] \leq 0$:
scenario.add_constraint("c1", "Margin", factor=3.0)
scenario.add_constraint("c2", "Margin", factor=3.0)

# %%
# We execute this scenario using the gradient-based optimizer SLSQP:
scenario.execute(algo_name="NLOPT_SLSQP", max_iter=200)

# %%
# and plot the optimization history:
scenario.post_process(post_name="OptHistoryView", save=True, show=False)

# %%
# Lastly,
# we can compare the numerical solution of this Sellar's MDO problem under uncertainty
result = scenario.optimization_result
(result.x_opt, result.constraint_values, result.f_opt)

# %%
# to the solution of the Sellar's MDO problem without uncertainty,
# namely
# $(x^*,c^*,f^*)=([0, 1.77, 0], [0, -20.58], 3.19)$.
