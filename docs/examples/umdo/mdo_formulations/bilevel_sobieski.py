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
r"""BiLevel applied to the SSBJ problem.

This example illustrates the use of the ``BiLevel`` formulation
to solve the
[Sobieski's SSBJ problem](https://gemseo.readthedocs.io/en/stable/problems/index.html#sobieski-s-ssbj-test-case)
under uncertainty.

The shared design variables are uncertain
and so are the objective and the constraints.
The uncertain objective is then made deterministic using the mean as a statistic
and the same for the constraints using margins of the form
"mean + twice standard deviation".
"""

from __future__ import annotations

from gemseo import configuration
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.algos.opt.nlopt.settings.nlopt_slsqp_settings import NLOPT_SLSQP_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario
from matplotlib import pyplot as plt
from numpy import atleast_2d

from gemseo_umdo.disciplines.utils import create_noising_discipline_chain
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configuration.fast = True

# %%
# ## Original discipline and spaces
#
# First,
# we create the disciplines:
mission = SobieskiMission()
structure = SobieskiStructure()
propulsion = SobieskiPropulsion()
aerodynamics = SobieskiAerodynamics()

# %%
# the design space:
design_space = SobieskiProblem().design_space

# %%
# an MDF-formulated scenario without uncertainties
mdf_scenario = MDOScenario(
    [aerodynamics, propulsion, structure, mission],
    "y_4",
    design_space,
    formulation_name="MDF",
    maximize_objective=True,
)
mdf_scenario.add_constraint("g_1", constraint_type="ineq")
mdf_scenario.add_constraint("g_2", constraint_type="ineq")
mdf_scenario.add_constraint("g_3", constraint_type="ineq")
# %%
# and solve it using the gradient-based SLSQP algorithm:
slsqp_settings = NLOPT_SLSQP_Settings(max_iter=100, ineq_tolerance=1e-3)
mdf_scenario.execute(algo_settings_model=slsqp_settings)

# %%
# In what follows,
# we will solve MDO problems under uncertainties
# from this uncertainty-free optimum $x_{\text{shared}}^*$:
x_opt_as_dict = mdf_scenario.get_result().optimization_result.x_opt_as_dict

# %%
# For that,
# we define the uncertain vector noising the shared design variables
# by means of a centered Gaussian vector
# with standard deviation $\sigma$ equal to $0.05x_{\text{shared}}^*/3$
# and restricted to the interval $[-3\sigma,3\sigma]$:
sigma = 0.05 * x_opt_as_dict["x_shared"] / 3
uncertain_space = ParameterSpace()
uncertain_space.add_random_vector(
    "u_x_shared",
    "OTNormalDistribution",
    sigma=sigma.tolist(),
    lower_bound=(-3 * sigma).tolist(),
    upper_bound=(3 * sigma).tolist(),
)

# %%
# Now,
# we can use the disciplines, design space and uncertain space
# to solve the Sobieski's SSBJ problem under uncertainty.
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
# from the uncertain-free optimum:
design_space = SobieskiProblem().design_space
for k, v in x_opt_as_dict.items():
    design_space.set_current_variable(k, v)
mdf_uscenario = UMDOScenario(
    [aerodynamics, propulsion, structure, mission],
    "y_4",
    design_space,
    uncertain_space,
    "Mean",
    Sampling_Settings(n_samples=n_samples, estimate_statistics_iteratively=False),
    maximize_objective=True,
    uncertain_design_variables={"x_shared": ("+", "u_x_shared")},
    formulation_name="MDF",
)
mdf_uscenario.add_constraint("g_1", "Margin")
mdf_uscenario.add_constraint("g_2", "Margin")
mdf_uscenario.add_constraint("g_3", "Margin")
# %%
# and solve it using the gradient-based SLSQP algorithm:
slsqp_settings = SLSQP_Settings(max_iter=max_iter, ineq_tolerance=1e-3)
mdf_uscenario.execute(algo_settings_model=slsqp_settings)

# %%
# We can see that
# the solution is more conservative than in the absence of uncertainty,
# with a smaller range and a smaller wing taper ratio ``"x_1[0]"``.
#
# ## BiLevel
#
# We solve the U-MDO problem using the ``BiLevel`` formulation
# from the same initial point:
design_space = SobieskiProblem().design_space
for k, v in x_opt_as_dict.items():
    design_space.set_current_variable(k, v)

# %%
# ### Noising disciplines chain
# In the case of the ``MDF`` formulation,
# we set ``uncertain_design_variables`` to ``{"x_shared": ("+", "u_x_shared")}``
# to rename the variable ``"x_shared"`` into ``"dv_x_shared"`` automatically.
# In this way,
# [UDMOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
# is able to propagate the uncertainty ``"u_x_shared"``
# through the multidisciplinary process
# using the expression
# $x_{\text{shared}}=dv_{x_{\text{shared}}}+u_{x_{\text{shared}}}$.
# Unfortunately,
# this automation cannot work for MDO formulations using sub-scenarios
# because their multidisciplinary processes do not know
# the variables ``"dv_x_shared"`` and ``"u_x_shared"``
# and cannot be modified.
# In this case,
# we cannot use the argument ``uncertain_design_variables``
# but we can create a chain of noising disciplines
# to be added to the disciplines used to create the sub-scenarios:
noising_discipline_chain = create_noising_discipline_chain(
    design_space, {"x_shared": ("+", "u_x_shared")}
)

# %%
# ### Sub-scenarios
#
# First,
# we create three sub-scenarios,
# namely the one to maximize the objective function
# with respect to the design variable specific to the aerodynamics
# using the gradient-based SLSQP algorithm:
scenario_aerodynamics = MDOScenario(
    [noising_discipline_chain, aerodynamics, mission],
    "y_4",
    design_space.filter("x_2", copy=True),
    formulation_name="MDF",
    maximize_objective=True,
)
scenario_aerodynamics.add_constraint("g_2", constraint_type="ineq")
scenario_aerodynamics.set_algorithm(algo_settings_model=slsqp_settings)
# %%
# the one to maximize the objective function
# with respect to the design variable specific to the propulsion
# using the gradient-based SLSQP algorithm:
scenario_propulsion = MDOScenario(
    [noising_discipline_chain, propulsion, mission],
    "y_4",
    design_space.filter("x_3", copy=True),
    formulation_name="MDF",
    maximize_objective=True,
)
scenario_propulsion.add_constraint("g_3", constraint_type="ineq")
scenario_propulsion.set_algorithm(algo_settings_model=slsqp_settings)
# %%
# and the one to maximize the objective function
# with respect to the design variable specific to the structure
# using the gradient-based SLSQP algorithm:
scenario_structure = MDOScenario(
    [noising_discipline_chain, structure, mission],
    "y_4",
    design_space.filter("x_1", copy=True),
    formulation_name="MDF",
    maximize_objective=True,
)
scenario_structure.add_constraint("g_1", constraint_type="ineq")
scenario_structure.set_algorithm(algo_settings_model=slsqp_settings)

# %%
# ### Main scenario
#
# Then,
# we create the main scenario from this sub-scenarios
# to maximize the mean of the objective function
# with respect to the global design variables ``"dv_x_shared"``
# and under equality constraints set as margins:
bilevel_uscenario = UMDOScenario(
    [scenario_aerodynamics, scenario_propulsion, scenario_structure],
    "y_4",
    design_space.filter("dv_x_shared", copy=True),
    uncertain_space,
    "Mean",
    Sampling_Settings(n_samples=n_samples),
    formulation_name="BiLevel",
    keep_opt_history=False,
    maximize_objective=True,
)
bilevel_uscenario.add_constraint("g_1", "Margin")
bilevel_uscenario.add_constraint("g_2", "Margin")
bilevel_uscenario.add_constraint("g_3", "Margin")

# %%
# and solve the MDO problem using the gradient-free COBYLA algorithm:
bilevel_uscenario.execute(
    algo_settings_model=NLOPT_COBYLA_Settings(max_iter=max_iter, ineq_tolerance=1e-3)
)

# %%
# ## Results
# In this last section,
# we will compare ``MDF`` and ``BiLevel`` in terms of results.
#
# First,
# we can have a look at the optimum solution in terms of global design variables:
mdf_x_opt = mdf_uscenario.optimization_result.x_opt
bilevel_x_opt = bilevel_uscenario.optimization_result.x_opt
# %%
# for ``MDF``:
mdf_x_opt[0]
# 0.29, 0.75, 0.75, 0.14, 0.06, 60000, 1.4, 2.5, 70, 1500

# %%
# for ``BiLevel``:
bilevel_x_opt[0]
# 0.06, 59762.5, 1.4, 2.5, 69.75, 1492.5

# %%
# The solutions are close in terms of global design variables.
#
# Then,
# we can have a look at the optimum solution in terms of mean objective:
mdf_f_opt = mdf_uscenario.optimization_result.f_opt
bilevel_f_opt = bilevel_uscenario.optimization_result.f_opt

# %%
# for ``MDF``:
mdf_f_opt
# -3844.6

# %%
# for ``BiLevel``:
bilevel_f_opt
# -3886.6

# %%
# The ``BiLevel`` solution seems to be better than the ``MDF`` one.
#
# Finally,
# we execute the ``BiLevel`` formulation at the bi-level optimum,
# taking care to save the samples with the option ``estimate_statistics_iteratively``:
bilevel_uscenario = UMDOScenario(
    [scenario_aerodynamics, scenario_propulsion, scenario_structure],
    "y_4",
    design_space.filter("dv_x_shared", copy=True),
    uncertain_space,
    "Mean",
    Sampling_Settings(n_samples=n_samples),
    formulation_name="BiLevel",
    maximize_objective=True,
    save_opt_history=False,
)
bilevel_uscenario.add_constraint("g_1", "Mean")
bilevel_uscenario.add_constraint("g_2", "Mean")
bilevel_uscenario.add_constraint("g_3", "Mean")
bilevel_uscenario.execute(algo_name="CustomDOE", samples=atleast_2d(bilevel_x_opt))

# %%
# which generates samples
# $\left(y_4\!\left(dv_{x_{\text{shared}}}^*,u_{x_{\text{shared}}}^{(i)}\right)\right)_{1\leq i \leq N}$:
database = bilevel_uscenario.formulation.mdo_formulation.optimization_problem.database
f_samples = -database.get_function_history("y_4").ravel()

# %%
# that we can plot using a histogram:
plt.boxplot(-f_samples, vert=False)
plt.plot(-mdf_f_opt, 1, "bo", label=r"$f^*(MDF)$")
plt.plot(-bilevel_f_opt, 1, "rs", label=r"$f^*(BiLevel)$")
plt.xlabel("f")
plt.grid()
plt.legend()
plt.savefig("bilevel_sobieski.png")

# %%
#
# ![ ](../../../../images/bilevel_sobieski.png)
#
# We can see that,
# on average,
# the ``BiLevel`` solution in red is better than the ``MDF`` solution in blue
# as mentioned above.
# On the other hand,
# there are realizations of $u_{x_{shared}}$ for which the solution could be even better.
# This is one of the advantages of using the ``BiLevel`` formulation under uncertainty:
# choosing the global design variable values now,
# and leave ourselves time to choose those of the local design variables,
# in the hope that, in the future,
# the uncertainties will decrease and lead us to an even more favorable solution.
#
# Finally,
# these results need to be refined,
# by adjusting the number of samples and by playing with the tolerances' thresholds.
