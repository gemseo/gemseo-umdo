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
"""# DOE"""

from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario

# %%
# Firstly,
# we define an [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline]
# implementing the random function $f(x,U)=(x+U)^2$:
discipline = AnalyticDiscipline({"y": "(x+u)**2"}, name="quadratic_function")

# %%
# where $x$ belongs to the interval $[-1,1]$:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)

# %%
# and $U$ is a standard Gaussian variable:
uncertain_space = ParameterSpace()
uncertain_space.add_random_variable("u", "OTNormalDistribution")

# %%
# Then,
# we build a [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
# to minimize a sampling-based estimation
# of the expectation $\mathbb{E}[Y]$ where $Y=f(x,U)$:
scenario = UDOEScenario(
    [discipline],
    "y",
    design_space,
    uncertain_space,
    "Mean",
    formulation_name="DisciplinaryOpt",
    statistic_estimation_settings=Sampling_Settings(n_samples=100),
)

# %%
# We execute it with a full-factorial design of experiments:
scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=100)

# %%
# and plot the history:
scenario.post_process(post_name="OptHistoryView", save=True, show=True)

# %%
# Notice that the numerical solution is close to $(x^*,f^*)=(0,1)$ as expected
# from the expression of the statistic: $\mathbb{E}[Y]=\mathbb{E}[(x+U)^2]=x^2+1$.
