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
r"""# A quantile constraint

In this example, we consider the optimization problem

$$
\begin{align}
&\min_{x\in[-3,3]} \mathbb{E}[(x+U)^2]\\
&\text{s.t. } Q_{0.95}[U-x]\leq 0
\end{align}
$$

where $U\sim\mathcal{N}(0,1)$ is a standard Gaussian variable,
$\mathbb{E}$ is the expectation operator
and $Q_{0.95}$ is the 0.95-[quantile](https://en.wikipedia.org/wiki/Quantile) operator.

The constraint means that $U\leq x$ with a probability of at least 0.95.
The objective can be rewritten as $x^2+1$
and the constraint as $x\geq q_{0.95}\approx 1.645$
where $q_{0.95}$ is the 0.95-quantile of the standard Gaussian distribution.
Then,
the solution is
$(x^*,\mathbb{E}[(x^*+U)^2])=(q_{0.95},q_{0.95}^2+1)\approx(1.645,3.706)$.
"""

from __future__ import annotations

from gemseo.discipline import AnalyticDiscipline
from gemseo.optimization import NLOPT_MMA_Settings
from gemseo.post import OptHistoryView_Settings
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings

from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

# %%
# Firstly,
# we define an [AnalyticDiscipline][gemseo.discipline.analytic.AnalyticDiscipline]
# computing the objective output $y=(x+U)^2$ and the constraint output $c=U-x$:
discipline = AnalyticDiscipline({"y": "(x+u)**2", "c": "u-x"})

# %%
# as well as the design space:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-3.0, upper_bound=3.0, value=2.5)

# %%
# and the uncertain space:
uncertain_space = RandomSpace()
uncertain_space.add_variable("u", OTNormalDistribution_Settings())

# %%
# Then,
# we define a [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
# estimating the statistics with a crude Monte Carlo sampling strategy
# with 1000 samples at each iteration of the optimization loop:
scenario = UMDOScenario(
    [discipline],
    design_space,
    uncertain_space,
    Sampling_Settings(n_samples=1000, estimate_statistics_iteratively=False),
)

# %%
# !!! note
#     By default,
#     the statistics are estimated iteratively,
#     i.e. without storing the samples.
#     As the quantile cannot be estimated iteratively,
#     we estimate the statistics from the whole set of samples.
scenario.add_objective("y", "Mean")

# %%
# We add the quantile constraint $Q_{0.95}[U-x]\leq 0$:
scenario.add_constraint("c", "Quantile", level=0.95)

# %%
# We also observe the median of the objective output:
scenario.add_observable("y", "Quantile", level=0.5)

# %%
# !!! note
#     For an objective or a constraint,
#     the level can be passed as $\alpha$ or $1-\alpha$ indifferently:
#     so that the statistic is conservative,
#     the level $\max(\alpha,1-\alpha)$ is used
#     for a minimized objective or a negativity constraint
#     and the level $\min(\alpha,1-\alpha)$
#     for a maximized objective or a positivity constraint.
#     For an observable, the level is used as is.
#
# We execute this scenario using the gradient-based optimizer MMA,
# which uses the analytical derivatives of the quantile:
scenario.execute(NLOPT_MMA_Settings(max_iter=100))

# %%
# and plot the optimization history:
scenario.post_process(OptHistoryView_Settings(save=False, show=True))

# %%
# Notice that the numerical solution
(scenario.optimization_result.x_opt[0], scenario.optimization_result.f_opt)

# %%
# is close to the theoretical solution
# $(x^*,\mathbb{E}[(x^*+U)^2])\approx(1.645,3.706)$,
# given the accuracy of the quantile estimator:
# with $N=1000$ samples,
# the standard error of the estimator of the 0.95-quantile of $U$
# is $\sqrt{0.95\times 0.05/N}/\phi(q_{0.95})\approx 0.067$
# where $\phi$ is the probability density function
# of the standard Gaussian distribution.
