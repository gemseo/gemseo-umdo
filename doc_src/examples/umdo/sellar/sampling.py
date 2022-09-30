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
Robust OPT - Sampling - Sellar problem
======================================
"""
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import configure_logger
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()

# %%
# Firstly,
# we instantiate the disciplines of the Sellar problem:
system = AnalyticDiscipline(
    {
        "obj": "x**2 + z2 + y1**2 + exp(-y2)",
        "c1": "3.16 - y1 ** 2",
        "c2": "y2 - 24.0",
    }
)
disc1 = AnalyticDiscipline({"y1": "(z1**2 + z2 + x - a*y2)**0.5"})
disc2 = AnalyticDiscipline({"y2": "abs(y1) + z1 + z2"})

# %%
# as well as the design space:
design_space = DesignSpace()
design_space.add_variable("x", 1, l_b=0.0, u_b=10.0, value=1.0)
design_space.add_variable("z1", 1, l_b=-10, u_b=10.0, value=4.0)
design_space.add_variable("z2", 1, l_b=0.0, u_b=10.0, value=3.0)

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
# we build an :class:`.UMDOScenario`
# to minimize a sampling-based estimation
# of the expectation :math:`\mathbb{E}[obj]`:
scenario = UMDOScenario(
    [system, disc1, disc2],
    "MDF",
    "obj",
    design_space,
    uncertain_space,
    "Mean",
    statistic_estimation="Sampling",
    statistic_estimation_parameters={"n_samples": 100},
)

# %%
# while satisfying margin constraints
# of the form :math:`\mathbb{E}[c_i]+3\mathbb{S}[c_i]`
scenario.add_constraint("c1", "Margin", factor=3.0)
scenario.add_constraint("c2", "Margin", factor=3.0)
# %%
# and execute it with a gradient-free optimizer:
scenario.execute({"algo": "NLOPT_COBYLA", "max_iter": 200})
scenario.post_process("OptHistoryView", save=True, show=False)
scenario.save_optimization_history("history.hdf5")
