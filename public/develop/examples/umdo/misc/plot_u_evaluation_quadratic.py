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
"""# Sampling statistics over the design space"""

from __future__ import annotations

from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import PYDOE_FULLFACT_Settings
from gemseo.post.dataset import YvsX
from gemseo.post.dataset.yvsx_settings import YvsX_Settings
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.u_evaluation_scenario import UEvaluationScenario

# %%
# Firstly,
# we define an [AnalyticDiscipline][gemseo.discipline.analytic.AnalyticDiscipline]
# implementing the random function $f(x,U)=(x+U)^2$:
discipline = AnalyticDiscipline({"y": "(x+u)**2"}, name="quadratic_function")

# %%
# where $x$ belongs to the interval $[-1,1]$:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)

# %%
# and $U$ is a standard Gaussian variable:
uncertain_space = RandomSpace()
uncertain_space.add_variable("u", OTNormalDistribution_Settings())

# %%
# Then,
# we build a
# [UEvaluationScenario][gemseo_umdo.scenarios.u_evaluation_scenario.UEvaluationScenario]
# to evaluate sampling-based estimations
# of the expectation $\mathbb{E}[Y]$ and the variance $\mathbb{V}[Y]$
# where $Y=f(x,U)$:
scenario = UEvaluationScenario(
    [discipline],
    design_space,
    uncertain_space,
    Sampling_Settings(n_samples=100),
)
scenario.add_observable("y", "Mean")
scenario.add_observable("y", "Variance")

# %%
# !!! note
#     Unlike
#     [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario],
#     this scenario has neither objective nor constraints.
#     Use a [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
#     with DOE settings
#     to sample a U-MDO problem.
#
# We execute it with a full-factorial design of experiments:
scenario.execute(PYDOE_FULLFACT_Settings(n_samples=20))

# %%
# export the results to a dataset:
dataset = scenario.to_dataset()
dataset

# %%
# and plot the estimation of the expectation versus the design variable:
plot = YvsX(dataset, settings=YvsX_Settings(x="x", y="E[y]"))
plot.execute(save=False, show=True)

# %%
# Notice that the estimation is close to the exact statistic
# $\mathbb{E}[Y]=\mathbb{E}[(x+U)^2]=x^2+1$.
