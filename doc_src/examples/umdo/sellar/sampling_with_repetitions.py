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
Robust OPT - Sampling with repetitions - Sellar problem
=======================================================
"""
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import configure_logger
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario
from matplotlib import pyplot as plt
from numpy import load
from numpy import save
from numpy import stack
from numpy import vstack

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
disc2 = AnalyticDiscipline({"y2": "2/10*log(1+exp(10*y1))-y1-2/10*log(2) + z1 + z2"})


# %%
# as well as a function to instantiate the design space:
def create_design_space() -> DesignSpace:
    """Create the design space for the Sellar problem."""
    design_space = DesignSpace()
    design_space.add_variable("x", 1, l_b=0.0, u_b=10.0, value=1.0)
    design_space.add_variable("z1", 1, l_b=-10, u_b=10.0, value=4.0)
    design_space.add_variable("z2", 1, l_b=0.0, u_b=10.0, value=3.0)
    return design_space


# %%
# Secondly,
# we define the uncertain space:
uncertain_space = ParameterSpace()
# %%
# with an uncertainty over the constant `"a"`:
uncertain_space.add_random_variable(
    "a", "OTTriangularDistribution", minimum=0.1, maximum=0.3, mode=0.2
)

# %
# Then,
# we build 10 :class:`.UMDOScenario`
# to minimize a sampling-based estimation
# of the expectation :math:`\mathbb{E}[obj]`
# and store the history of the design values:
x_hist = []
for i in range(10):
    print(i)
    scenario = UMDOScenario(
        [system, disc1, disc2],
        "MDF",
        "obj",
        create_design_space(),
        uncertain_space,
        "Mean",
        statistic_estimation="Sampling",
        statistic_estimation_parameters={
            "algo": "OT_LHS",
            "n_samples": 100,
            "seed": i + 1,
        },
    )
    scenario.add_constraint("c1", "Margin", factor=3.0)
    scenario.add_constraint("c2", "Margin", factor=3.0)
    scenario.execute({"algo": "NLOPT_COBYLA", "max_iter": 100})
    x_hist.append(vstack(scenario.formulation.opt_problem.database.get_x_history()))

# %%
# Lastly,
# we plot the variability of the optimization history with boxplots:
print(x_hist)
x_hist = stack(x_hist)
save("x_hist.npy", x_hist)
x_hist = load("x_hist.npy")
plt.boxplot(x_hist[:, :, 0])
plt.savefig("hist.png")
