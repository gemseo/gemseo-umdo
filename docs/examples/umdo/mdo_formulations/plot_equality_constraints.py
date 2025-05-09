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
r"""# Managing uncertain equality constraints

This example shows a simple way
to manage an uncertain equality constraint associated with a threshold,
by forcing the mean of its left-hand side to be equal to this threshold
and its variance to be zero.

The reference optimization problem consists in
minimizing the Rosenbrock function $f(x,y)=(1-x)^2+100(y-x^2)^2$ over $[-2,2]^2$
under the equality constraint $h(x,y)=r^2$ with $h(x,y)=(x-1)^2+(y-1)^2$ and $r=0.25$.

In the following,
we suppose that $f(x,y)$ and $h(x,y)$ depend on uncertain parameters $a$ and $b$,
as $f(x,y)=(a-x)^2+100(y-x^2)^2$ and $h(x,y)=(x-b)^2+(y-1)^2$,
and seek to minimize $\mathbb{E}[f(x,y)]$
under the equality constraints $\mathbb{E}[h(x,y)]=r^2$ and $\mathbb{V}[h(x,y)]=0$.
"""

from gemseo import configure_logger
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.mdo_scenario import MDOScenario
from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from numpy import array

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()

# %%
# ## Discipline and design space
#
# First,
# we create a discipline to evaluate $f$ and $h$
# from the design variables $x$ and $y$
# and from the uncertain variables $a$ and $b$ initialized at 1:
discipline = AnalyticDiscipline({
    "f": "(a-x)**2+100*(y-x**2)**2",
    "h": "(x-b)**2+(y-1)**2",
})
discipline.io.input_grammar.defaults["a"] = array([1.0])
discipline.io.input_grammar.defaults["b"] = array([1.0])

# %%
# Then,
# we create the design space:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2, upper_bound=2.0)
design_space.add_variable("y", lower_bound=-2, upper_bound=2.0)
# %%
# and the initial design point:
initial_design = array([1.75, 1.75])

# %%
# For visualization purposes,
# we sample the objective function over a regular grid:
scenario = MDOScenario(
    [discipline],
    "f",
    design_space,
    formulation_name="DisciplinaryOpt",
)
scenario.execute(algo_name="OT_FULLFACT", n_samples=20 * 20)
# %%
# and store the 400 samples:
samples = scenario.to_dataset()

# %%
# ## Constrained optimization problem
#
# Then,
# we define the uncertainty-free constrained optimization problem:
radius = 0.25
scenario = MDOScenario(
    [discipline],
    "f",
    design_space,
    formulation_name="DisciplinaryOpt",
)
scenario.add_constraint("h", value=radius**2)
# %%
# and solve it using the gradient-based SLSQP algorithm:
scenario.execute(algo_name="SLSQP", max_iter=100)
x_opt = scenario.optimization_result.x_opt

# %%
# ## Constrained optimization problem under uncertainty
#
# Lastly,
# we create the constrained optimization problem under uncertainty.
#
# ### Uncertain space
#
# First,
# we need to define the uncertain space
# with independent normal variables centered at 1 with standard deviation equal to 1/6:
uncertain_space = ParameterSpace()
uncertain_space.add_random_variable("a", "OTNormalDistribution", mu=1.0, sigma=1 / 6)
uncertain_space.add_random_variable("b", "OTNormalDistribution", mu=1.0, sigma=1 / 6)

# %%
# ### Problem
#
# Then,
# we reset the design space to the initial solution
design_space.set_current_value(initial_design)
# %%
# and create the scenario
# by forcing the mean of $h(x,y)$ to be equal to $r^2$ and its variance to be zero:
scenario = UMDOScenario(
    [discipline],
    "f",
    design_space,
    uncertain_space,
    "Mean",
    Sampling_Settings(n_samples=100, estimate_statistics_iteratively=False),
    formulation_name="DisciplinaryOpt",
)
scenario.add_constraint("h", "Mean", constraint_type="eq", value=radius**2)
scenario.add_constraint("h", "Variance", constraint_type="eq")
# %%
# Finally,
# we solve this optimization problem using the gradient-based SLSQP algorithm:
scenario.execute(algo_name="SLSQP", max_iter=100)

# %%
# ## Results
# In this last section,
# we plot and analyze the results.
fig, ax = plt.subplots()
# The Rosenbrock function plotted as filled contours:
ax.contourf(
    samples.get_view(variable_names="x").to_numpy().reshape((20, 20)),
    samples.get_view(variable_names="y").to_numpy().reshape((20, 20)),
    samples.get_view(variable_names="f").to_numpy().reshape((20, 20)),
    levels=20,
    cmap=colormaps["Greys"],
)
# The initial design solution:
ax.plot(*initial_design, "dk", label="Initial solution")
# The solution of the uncertainty-free unconstrained optimization problem:
ax.plot(1.0, 1.0, "*k", label="argmin f(a=1,x,y)")
# The solution of the uncertainty-free constrained optimization problem:
ax.plot(
    *x_opt,
    "ob",
    label="argmin f(a=1,x,y) s.t. h(b=1,x,y)=r²",
)
# The level set associated with the equality constraint:
ax.add_patch(plt.Circle((1.0, 1.0), radius, fill=False, label="x,y s.t. h(b=1,x,y)=r²"))
# The solution of the constrained optimization problem under uncertainty:
ax.plot(
    *scenario.optimization_result.x_opt,
    "sr",
    label="argmin E[f(a,x,y)] s.t. E[h(b,x,y)]=r² and V[h(b,x,y)]=0",
)
ax.set_aspect("equal", adjustable="box")
plt.legend()
plt.show()
# %%
# We can see that
# the uncertainty-free and uncertainty-based optima are close but different.
# We can also note that
# the solution under uncertainty is unfortunately not feasible.
# This could be corrected by better tuning the statistics estimation algorithm
# or by changing the optimization algorithm.
# However,
# this is beyond the scope of this example,
# the aim of which is to show a simple way of dealing with equality constraints.
