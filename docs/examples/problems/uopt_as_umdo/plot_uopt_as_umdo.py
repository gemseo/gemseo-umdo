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
r"""
# Make a monodisciplinary optimization under uncertainty problem multidisciplinary

## Introduction

[UOptAsUMDOScenario][gemseo_umdo.problems.uopt_as_umdo_scenario.UOptAsUMDOScenario]
is a monodisciplinary optimization scenario under uncertainty made multidisciplinary.
The only requirement is that
the discipline has at least three scalar inputs defined as design variables,
and at least one output defined as an objective.
These design variables may be uncertain.
The discipline may also have other inputs,
defined as design or uncertain variables,
and outputs,
defined as objectives or constraints.
This scenario can be used to enrich a catalog of benchmark U-MDO problems,
based on the observation that
MDO benchmark problems are far less numerous than optimization problems,
especially in the case of MDO under uncertainty.

This example illustrates it
in the case of the minimization of the 3-dimensional Rosenbrock function

$$\mathbb{E}[f(z,U,V)] = \mathbb{E}[100(z_2-(Uz_1)^2)^2 + (1-Vz_1)^2 + 100(z_1-(Uz_0)^2)^2 + (1-Vz_0)^2]$$

over the hypercube $[-1,1]^3$.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import generate_coupling_graph
from gemseo.algos.parameter_space import ParameterSpace
from numpy import array

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.problems.uopt_as_umdo_scenario import UOptAsUMDOScenario
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()

# %%
# ## Discipline and spaces
# First,
# we create the discipline implementing the Rosenbrock function:
discipline = create_discipline(
    "AnalyticDiscipline",
    expressions={
        "f": "100*(z_2-(u*z_1)**2)**2+(1-v*z_1)**2+100*(z_1-(u*z_0)**2)**2+(1-v*z_0)**2"
    },
    name="Rosenbrock",
)
# %%
# as well as the design space:
design_space = create_design_space()
design_space.add_variable("z_0", lower_bound=-1, upper_bound=1)
design_space.add_variable("z_1", lower_bound=-1, upper_bound=1)
design_space.add_variable("z_2", lower_bound=-1, upper_bound=1)
# %%
# and the uncertain space:
uncertain_space = ParameterSpace()
uncertain_space.add_random_variable("u", "OTNormalDistribution", mu=1.0, sigma=0.01)
uncertain_space.add_random_variable("v", "OTNormalDistribution", mu=1.0, sigma=0.01)
# %%
# We choose $x^{(0)}=(-0.25, 0.75, -0.9)$
# as the starting point of the optimization:
initial_point = array([-0.25, 0.75, -0.9])
design_space.set_current_value(initial_point)

# %%
# ## Optimization problem under uncertainty
# Then,
# we define the optimization problem under uncertainty,
# with statistics estimated by sampling:
u_opt_scenario = UMDOScenario(
    [discipline],
    "f",
    design_space,
    uncertain_space,
    "Mean",
    Sampling_Settings(n_samples=50, estimate_statistics_iteratively=False),
    formulation_name="DisciplinaryOpt",
)
# %%
# and solve it using the SLSQP algorithm:
u_opt_scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
# %%
# ## MDO problem under uncertainty
# Now,
# we use the
# [UOptAsUMDOScenario][gemseo_umdo.problems.uopt_as_umdo_scenario.UOptAsUMDOScenario]
# to rewrite this optimization problem under uncertainty
# as an MDO problem under uncertainty with two strongly coupled disciplines.
#
# First,
# we reset the design space to the initial point:
design_space.set_current_value(initial_point)
# %%
# and create the
# [UOptAsUMDOScenario][gemseo_umdo.problems.uopt_as_umdo_scenario.UOptAsUMDOScenario],
# orchestrated by an MDF formulation:
umdo_scenario = UOptAsUMDOScenario(
    discipline,
    "f",
    design_space,
    uncertain_space,
    "Mean",
    Sampling_Settings(n_samples=50, estimate_statistics_iteratively=False),
    formulation_name="MDF",
)
# %%
# Then,
# we can see that the design variables have been renamed:
design_space
# %%
# This renaming is based on the convention:
#
# - the first design variable is the global design variable and is named $x_0$,
# - the $(1+i)$-th design variable is the local design variable
#   specific to the $i$-th strongly coupled discipline
#   and is named $x_{1+i}$.
#
# We can also have a look to the coupling graph:
generate_coupling_graph(umdo_scenario.disciplines, file_path="")
# %%
# and see that there are two strongly coupled disciplines $D_1$ and $D_2$,
# connected by the coupling variables $y_1$ and $y_2$.
# These disciplines are weakly coupled to a downstream link discipline $L$,
# which is weakly coupled to the downstream original discipline.
# Let us note that the link discipline computes
# the values of the design variables in the original optimization problem
# from the values of the design and coupling variables in the MDO problem.
#
# We can also represent the MDO process using an XDSM:
umdo_scenario.xdsmize(save_html=False, pdf_build=False)
# %%
# Lastly,
# we solve this scenario using the SLSQP algorithm:
umdo_scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
# %%
# We can see that
# the numerical solution corresponds to the one found in the monodisciplinary case.
