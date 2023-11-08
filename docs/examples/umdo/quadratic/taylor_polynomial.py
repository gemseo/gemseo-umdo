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
# Robust OPT - Taylor polynomial - Quadratic function
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline

from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

configure_logger()

# %%
# Firstly,
# we define an [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline]
# implementing the random function $f(x,U)=(x+U)^2$:
discipline = AnalyticDiscipline({"y": "(x+u)**2"}, name="quadratic_function")

# %%
# where $x$ belongs to the interval $[-1,1]$:
design_space = DesignSpace()
design_space.add_variable("x", l_b=-1, u_b=1.0, value=0.5)

# %%
# and $U$ is a standard Gaussian variable:
uncertain_space = ParameterSpace()
uncertain_space.add_random_variable("u", "OTNormalDistribution")

# %%
# Then,
# we build a [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
# to minimize a linearization-based estimation
# of the expectation $\mathbb{E}[Y]$ where $Y=f(x,U)$:
scenario = UMDOScenario(
    [discipline],
    "DisciplinaryOpt",
    "y",
    design_space,
    uncertain_space,
    "Mean",
    statistic_estimation="TaylorPolynomial",
)

# %%
# We execute it with a gradient-free optimizer:
scenario.execute({"algo": "NLOPT_COBYLA", "max_iter": 100})

# %%
# and plot the history:
scenario.post_process("OptHistoryView", save=False, show=True)

# %%
# Notice that the numerical solution is close to $(x^*,f^*)=(0,1)$ as expected
# from the expression of the statistic: $\mathbb{E}[Y]=\mathbb{E}[(x+U)^2]=x^2+1$.
