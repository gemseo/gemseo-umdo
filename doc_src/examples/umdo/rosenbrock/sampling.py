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
Robust OPT - Sampling - Rosenbrock function
===========================================
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
# we define an :class:`.AnalyticDiscipline`
# implementing a random version of the Rosenbrock function
# :math:`f(x,y,U)=(U-x)^2+100(y-x^2)^2`:
discipline = AnalyticDiscipline({"z": "(u-x)**2+100*(y-x**2)**2"}, name="Rosenbrock")

# %%
# where :math:`x,y` belongs to the interval :math:`\[-2,2\]`:
design_space = DesignSpace()
design_space.add_variable("x", l_b=-2, u_b=2.0, value=-2.0)
design_space.add_variable("y", l_b=-2, u_b=2.0, value=-2.0)

# %%
# and :math:`U` is a Gaussian variable with unit mean
# and standard deviation equal to 0.05:
uncertain_space = ParameterSpace()
uncertain_space.add_random_variable("u", "OTNormalDistribution", mu=1.0, sigma=0.05)

# %%
# Then,
# we build a :class:`UMDOScenario` to minimize a sampling-based estimation
# of the expectation :math:`\mathbb{E}[Y]` where :math:`Y=f(x,y,U)`:
scenario = UMDOScenario(
    [discipline],
    "DisciplinaryOpt",
    "z",
    design_space,
    uncertain_space,
    "Mean",
    statistic_estimation="Sampling",
    statistic_estimation_parameters={"n_samples": 10},
)

# %%
# and execute it with a gradient-based optimizer:
scenario.set_differentiation_method("finite_differences")
# %%
# .. note::
#    The statistics do not allow for the moment to use analytical derivatives.
#    Please use finite differences or complex step to approximate the gradients.
scenario.execute({"algo": "NLOPT_SLSQP", "max_iter": 100})

# %%
# Lastly,
# we can plot the optimization history view:
scenario.post_process("OptHistoryView", save=True, show=True)
