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
"""# Control variate vs Sampling"""

from __future__ import annotations

import numpy as np
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.string_tools import MultiLineString
from matplotlib import pyplot as plt
from numpy import array
from numpy import ndarray
from numpy import quantile
from scipy.spatial.distance import cdist

from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

# %%
# Firstly,
# we define an [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline]
# implementing a random version of the Rosenbrock function
# $f(x,y,U)=(U-x)^2+100(y-x^2)^2$:
discipline = AnalyticDiscipline({"z": "(a-x)**2+100*(y-x**2)**2"}, name="Rosenbrock")

# %%
# where $x,y$ belongs to the interval $[-2,2]$:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2, upper_bound=2.0, value=-2.0)
design_space.add_variable("y", lower_bound=-2, upper_bound=2.0, value=-2.0)

# %%
# and $U$ is a Gaussian variable with unit mean
# and standard deviation equal to 0.05:
uncertain_space = ParameterSpace()
uncertain_space.add_random_variable("a", "OTNormalDistribution", mu=1.0, sigma=0.05)

# %%
# Then,
# we want to build a [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
# to minimize a sampling-based estimation
# of the expectation $\mathbb{E}[Y]$ where $Y=f(x,y,U)$:
# For that,
# we compare an approach based on crude Monte Carlo
# and an approach based on a linearized model as control variate
# and repeat it 20 times to get statistics on the results:
method_to_x_opt = {"Sampling": [], "ControlVariate": []}
for i in range(20):
    for method in ["Sampling", "ControlVariate"]:
        scenario = UMDOScenario(
            [discipline],
            "DisciplinaryOpt",
            "z",
            design_space,
            uncertain_space,
            "Mean",
            statistic_estimation=method,
            statistic_estimation_parameters={
                "algo": "OT_MONTE_CARLO",
                "n_samples": 10,
                "seed": i + 1,
            },
        )
        scenario.set_differentiation_method("finite_differences")
        scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
        method_to_x_opt[method].append(scenario.optimization_result.x_opt.tolist())


# %%
# Lastly,
# we print and plot the comparison
# in terms of distance to the theoretical solution $x^*=(1,1)$:
def ecdf(data: ndarray) -> tuple[ndarray, ndarray]:
    """Empirical cumulative distribution function.

    Args:
        data: The data.

    Returns:
        The quantiles and the cumulative probabilities.
    """
    quantiles, counts = np.unique(data, return_counts=True)
    return quantiles, np.cumsum(counts).astype(np.double) / data.size


comparison = MultiLineString()
for index, method in enumerate(["Sampling", "ControlVariate"]):
    distances_to_one = cdist(array(method_to_x_opt[method]), array([[1.0, 1.0]]))
    x, y = ecdf(abs(distances_to_one))
    plt.plot(x, y, "-" * index, label=method)
    comparison.add(method)
    comparison.indent()
    comparison.add(f"Mean: {distances_to_one.mean():.2e}")
    comparison.add(f"Standard deviation: {distances_to_one.std():.2e}")
    comparison.add(f"0.05-quantile: {quantile(distances_to_one, 0.05):.2e}")
    comparison.add(f"0.95-quantile: {quantile(distances_to_one, 0.95):.2e}")
    comparison.dedent()

print(comparison)

plt.xlabel("Distance to the theoretical solution x=(1,1)")
plt.ylabel("Cumulative distribution function")
plt.legend()
plt.show()
