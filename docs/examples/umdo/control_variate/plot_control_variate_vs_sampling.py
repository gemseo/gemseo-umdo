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
r"""# Control variate vs Sampling

In this example,
we will compare the performance of the `Sampling` and `ControlVariate` techniques
to estimate statistics in an MDO problem under uncertainty.
For that purpose,
we consider a version of the Rosenbrock problem under uncertainty:

$$\min_{x_1,x_2\in[-1,2]} \mathbb{E}[f(x_1,x_2,U)]$$

where $f(x_1,x_2,u)=(u-x_1)^2+100(x_2-x_1^2)^2$
and $U$ is a Gaussian random variable with mean 1 and standard deviation 0.1.

Note that this problem can be rewritten as

$$\min_{x_1,x_2\in[-1,2]} \sigma^2 + \tilde{f}(x_1,x_2)$$

where $\tilde{f}(x_1,x_2)=f(x_1,x_2,1)$ is the standard Rosenbrock function.
Therefore,
the analytical solution is $(x^*,f(x_1^*,x_2^*))=((1,1),0)$.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.datasets.dataset import Dataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.post.dataset.boxplot import Boxplot
from gemseo.settings.doe import OT_MONTE_CARLO_Settings
from numpy import array
from numpy.linalg import norm

from gemseo_umdo.formulations.control_variate_settings import ControlVariate_Settings
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

if TYPE_CHECKING:
    from gemseo.typing import RealArray

    from gemseo_umdo.formulations.base_umdo_formulation_settings import (
        BaseUMDOFormulationSettings,
    )

# %%
# First,
# we define the discipline
# to compute the output $z=f(x_1,x_2,u)$ from the inputs $x_1$, $x_2$ and $u$:
discipline = AnalyticDiscipline({"z": "(u-x1)**2+100*(x2-x1**2)**2"}, name="f")

# %%
# as well as the design space $[-2,2]^2$:
design_space = DesignSpace()
design_space.add_variable("x1", lower_bound=-2, upper_bound=2.0, value=-2.0)
design_space.add_variable("x2", lower_bound=-2, upper_bound=2.0, value=-2.0)

# %%
# and the uncertain space:
uncertain_space = ParameterSpace()
sigma = 0.1
uncertain_space.add_random_variable("u", "OTNormalDistribution", mu=1.0, sigma=sigma)


# %%
# Then,
# we create a function to solve the optimization problem under uncertainty
# from a statistics estimation technique
# and a seed for the pseudo-random numbers generator:
def solve_problem(
    settings_class: type[BaseUMDOFormulationSettings], seed: int
) -> RealArray:
    """Solve the optimization problem under uncertainty.

    Args:
        settings_class: The class
            for defining the settings of the statistics estimation technique.
        seed: The seed for the pseudo-random numbers generator.

    Returns:
        The optimal values of the design variables.
    """
    discipline.cache.clear()
    scenario = UMDOScenario(
        [discipline],
        "z",
        design_space,
        uncertain_space,
        "Mean",
        formulation_name="DisciplinaryOpt",
        statistic_estimation_settings=settings_class(
            doe_algo_settings=OT_MONTE_CARLO_Settings(n_samples=50, seed=i)
        ),
    )
    scenario.set_differentiation_method("finite_differences")
    scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
    return scenario.optimization_result.x_opt


# %%
# Now,
# we are ready to solve the optimization problem under uncertainty
# with the `Sampling` and `ControlVariate` techniques
# and repeat this experiment 10 times to account for samples variability:
x_opt_s = []
x_opt_cv = []
x_star = array([1.0, 1.0])
for i in range(20):
    x_opt = solve_problem(Sampling_Settings, i)
    x_opt_s.append([norm(x_opt - x_star)])
    x_opt = solve_problem(ControlVariate_Settings, i)
    x_opt_cv.append([norm(x_opt - x_star)])

# %%
# In these lines,
# the point `x_star` represents the analytical solution $x^*$
# while `x_opt_method[i]` represents the numerical solution obtained
# with the method `Sampling` (s) or `ControlVariate` (cv)
# at the `i`-th repetition.
#
# Finally,
# we use boxplots to compare the `Sampling` and `ControlVariate` techniques
# in terms of estimation error
# by looking not only at the average value but also at the variability.
dataset_s = Dataset()
dataset_s.add_variable("x_opt", x_opt_s)
dataset_s.name = "Sampling"

dataset_cv = Dataset()
dataset_cv.add_variable("x_opt", x_opt_cv)
dataset_cv.name = "Control variate"

# %%
# Below are the boxplots
# showing the estimation error in the Euclidean norm for the optimal design:
boxplot = Boxplot(dataset_s, dataset_cv, variables=["x_opt"])
boxplot.execute(save=False, show=True)

# %%
# We can see that the results are more accurate with the `ControlVariate` technique
# even though the calculation budget is the same.
#
# To conclude,
# the [control variates](https://en.wikipedia.org/wiki/Control_variates) method
# is a powerful variance reduction technique
# and its combination with surrogate models,
# such as first-order Taylor polynomial in this example,
# can facilitate its adoption in many contexts.
