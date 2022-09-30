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
"""
Robust MDO - Sampling - Sobieski
================================
"""
from __future__ import annotations

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import configure_logger
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()

# %%
# Firstly,
# we instantiate the discipline of Sobieski's SSBJ problem:
mission = SobieskiMission()
structure = SobieskiStructure()
propulsion = SobieskiPropulsion()
aerodynamics = SobieskiAerodynamics()

# %%
# as well as the design space:
design_space = SobieskiProblem().design_space

# %%
# Secondly,
# we define the uncertain space:
uncertain_space = ParameterSpace()
# %%
# with an uncertainty over the constant `"c_4"`:
uncertain_space.add_random_variable(
    "c_4", "OTNormalDistribution", mu=0.01375, sigma=0.01375 * 0.05
)
# %%
# and an uncertainty over the design variable `"x_2"`,
# expressed as an additive term `"u_x_2"`
# defined just after in the :class:`.UMDOScenario`:
uncertain_space.add_random_variable(
    "u_x_2", "OTNormalDistribution", mu=0.0, sigma=1 * 0.05
)

# %
# Then,
# we build an :class:`.UMDOScenario`
# to maximize a sampling-based estimation
# of the expectation :math:`\mathbb{E}[y_4]`:
scenario = UMDOScenario(
    [mission, structure, propulsion, aerodynamics],
    "MDF",
    "y_4",
    design_space,
    uncertain_space,
    "Mean",
    statistic_estimation="Sampling",
    statistic_estimation_parameters={"n_samples": 10},
    maximize_objective=True,
    uncertain_design_variables={"x_2": "{}+u_x_2"},
)

# %
# while satisfying margin constraints
# of the form :math:`\mathbb{E}[g_i]+3\mathbb{S}[g_i]`
scenario.add_constraint("g_1", "Margin", factor=3.0)
scenario.add_constraint("g_2", "Margin", factor=3.0)
scenario.add_constraint("g_3", "Margin", factor=3.0)

# %%
# and execute it with a gradient-free optimizer:
scenario.execute({"algo": "NLOPT_COBYLA", "max_iter": 100})

# %%
# Lastly,
# we can plot the optimization history view:
scenario.post_process("OptHistoryView", save=False, show=True)
