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
r"""BiLevel applied to the Rosenbrock problem.

This example illustrates the use of the ``BiLevel`` formulation
to solve the 3-dimensional Rosenbrock problem under uncertainty:

$$\mathbb{E}[f(z,U,V)] = \mathbb{E}[100(z_2-(Uz_1)^2)^2 + (1-Vz_1)^2 + 100(z_1-(Uz_0)^2)^2 + (1-Vz_0)^2]$$

relative to $z$ in the hypercube $[-1,1]^3$,
where $\mathbb{E}$ is the expectation operator
and $U$ and $V$ are independent random variables
normally distributed
with mean equal to 1 and standard deviation equal to 0.01[@AzizAlaoui2025].

We use the
[create_disciplines][gemseo.problems.mdo.opt_as_mdo_scenario.create_disciplines]
function
to make this optimization multidisciplinary.
"""

from __future__ import annotations

from gemseo import configure
from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_parameter_space
from gemseo.problems.mdo.opt_as_mdo_scenario import create_disciplines
from gemseo.scenarios.mdo_scenario import MDOScenario
from matplotlib import pyplot as plt
from numpy import array
from numpy import atleast_2d

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()
configure(False, False, True, False, False, False, False)

# %%
# ## Original discipline and spaces
#
# First,
# we create the Rosenbrock discipline:
rosenbrock = create_discipline(
    "AnalyticDiscipline",
    expressions={
        "f": "100*(z_2-(u*z_1)**2)**2+(1-v*z_1)**2+100*(z_1-(u*z_0)**2)**2+(1-v*z_0)**2"
    },
    name="Rosenbrock",
)

# %%
# the design space:
design_space = create_design_space()
design_space.add_variable("z_0", lower_bound=-1, upper_bound=1)
design_space.add_variable("z_1", lower_bound=-1, upper_bound=1)
design_space.add_variable("z_2", lower_bound=-1, upper_bound=1)

# %%
# and the uncertain space:
uncertain_space = create_parameter_space()
uncertain_space.add_random_variable("u", "OTNormalDistribution", mu=1.0, sigma=0.01)
uncertain_space.add_random_variable("v", "OTNormalDistribution", mu=1.0, sigma=0.01)

# %%
# Then,
# we create the disciplines of the multidisciplinary problem:
disciplines = create_disciplines(rosenbrock, design_space)

# %%
# which are the Rosenbrock discipline,
# a linear link discipline and two strongly coupled linear disciplines:
rosenbrock, link_discipline, discipline_1, discipline_2 = disciplines

# %%
# Now,
# we can use these disciplines, design space and uncertain space
# to solve the Rosenbrock problem under uncertainty in a multidisciplinary way.
# More precisely,
# we will solve the U-MDO problem
# not only with the ``BiLevel`` formulation but also with the ``MDF`` one
# to show the difference in nature between these formulations
# and highlight the characteristics of ``BiLevel``.
#
# The optimization algorithms will use a maximum of 100 iterations
# and the expectation will be estimated from 30 samples:
max_iter = 100
n_samples = 30

# %%
# ## MDF
#
# We solve the U-MDO problem using the ``MDF`` formulation
# from the initial point $x=(-0.25, 0.75, -0.9)$:
initial_point = array([-0.25, 0.75, -0.9])
design_space.set_current_value(initial_point)
mdf_scenario = UMDOScenario(
    [rosenbrock, link_discipline, discipline_1, discipline_2],
    "f",
    design_space,
    uncertain_space,
    "Mean",
    Sampling_Settings(n_samples=n_samples, estimate_statistics_iteratively=False),
    formulation_name="MDF",
)
# %%
# and solve it using the gradient-based SLSQP algorithm:
mdf_scenario.execute(algo_name="NLOPT_SLSQP", max_iter=max_iter)

# %%
# ## BiLevel
#
# We solve the U-MDO problem using the ``BiLevel`` formulation
# from the initial point $x=(-0.25, 0.75, -0.9)$:
design_space.set_current_value(initial_point)
# %%
# ### Sub-scenarios
#
# First,
# we create two sub-scenarios,
# namely the one to minimize the objective function
# with respect to the local design variable $x_1$
# using the gradient-based SLSQP algorithm:
scenario_1 = MDOScenario(
    [discipline_1, link_discipline, rosenbrock],
    "f",
    design_space.filter("x_1", copy=True),
    formulation_name="DisciplinaryOpt",
)
scenario_1.set_algorithm(algo_name="SLSQP", max_iter=max_iter)
# %%
# namely the one to minimize the objective function
# with respect to the local design variable $x_2$
# using the gradient-based SLSQP algorithm:
scenario_2 = MDOScenario(
    [discipline_2, link_discipline, rosenbrock],
    "f",
    design_space.filter("x_2", copy=True),
    formulation_name="DisciplinaryOpt",
)
scenario_2.set_algorithm(algo_name="SLSQP", max_iter=max_iter)
# %%
# ### Main scenario
#
# Then,
# we create the main scenario from this sub-scenarios
# to minimize the objective function with respect to the global design variable $x_0$:
bilevel_scenario = UMDOScenario(
    [scenario_1, scenario_2, rosenbrock],
    "f",
    design_space.filter("x_0", copy=True),
    uncertain_space,
    "Mean",
    Sampling_Settings(n_samples=n_samples, estimate_statistics_iteratively=False),
    formulation_name="BiLevel",
    save_opt_history=False,
)
# %%
# and solve the MDO problem using the gradient-free COBYLA algorithm:
bilevel_scenario.execute(algo_name="NLOPT_COBYLA", max_iter=max_iter)

# %%
# ## Results
# In this last section,
# we will compare ``MDF`` and ``BiLevel`` in terms of results.
#
# First,
# we can have a look at the optimum solution in terms of global variables:
mdf_x_opt = mdf_scenario.optimization_result.x_opt
bilevel_x_opt = bilevel_scenario.optimization_result.x_opt
# %%
# for ``MDF``:
mdf_x_opt[0]
# %%
# for ``BiLevel``:
bilevel_x_opt[0]

# %%
# The solutions are very close in terms of global design variables.
#
# Then,
# we can have a look at the optimum solution in terms of mean objective:
mdf_f_opt = mdf_scenario.optimization_result.f_opt
bilevel_f_opt = bilevel_scenario.optimization_result.f_opt

# %%
# for ``MDF``:
mdf_f_opt

# %%
# for ``BiLevel``:
bilevel_f_opt

# %%
# The ``BiLevel`` solution seems to be better than the ``MDF`` one.
#
# Finally,
# we execute the ``BiLevel`` formulation at the bi-level optimum:
bilevel_scenario.execute(algo_name="CustomDOE", samples=atleast_2d(bilevel_x_opt))

# %%
# which generates samples $\left(f(x_0^*,U^{(i)},V^{(i)})\right)_{1\leq i \leq N}$:
database = bilevel_scenario.formulation.mdo_formulation.optimization_problem.database
f_samples = database.get_function_history("f").ravel()

# %%
# that we can plot using a histogram:
plt.boxplot(f_samples, vert=False)
plt.plot(mdf_f_opt, 1, "bo", label=r"$f^*(MDF)$")
plt.plot(bilevel_f_opt, 1, "rs", label=r"$f^*(BiLevel)$")
plt.xlabel("f")
plt.grid()
plt.legend()
plt.show()

# %%
# We can see that,
# on average,
# the ``BiLevel`` solution in red is better than the ``MDF`` solution in blue.
# On the other hand,
# there are values of $U$ and $V$ for which the solution could be even better.
# This is one of the advantages of using the ``BiLevel`` formulation under uncertainty:
# choosing the global design variable values now,
# and leave ourselves time to choose those of the local design variables,
# in the hope that, in the future,
# the uncertainties will decrease and lead us to an even more favorable solution.
