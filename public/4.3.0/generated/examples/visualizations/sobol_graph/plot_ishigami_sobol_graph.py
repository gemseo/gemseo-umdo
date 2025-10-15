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
r"""# Sobol' graph for the Ishigami use case.

This example illustrates
the concept of [SobolGraph][gemseo_umdo.visualizations.sobol_graph.SobolGraph]
in the case of the Ishigami problem:

$$f(X_1,X_2,X_3) = \sin(X_1)+ 7\sin(X_2)^2 + 0.1X_3^4\sin(X_1)$$

where $X_1$, $X_2$ and $X_3$ are independent random variables
uniformly distributed over the interval $[-\pi,\pi]$.

We are going to represent the Sobol' indices of $f$
using a [SobolGraph][gemseo_umdo.visualizations.sobol_graph.SobolGraph].
"""

from __future__ import annotations

from gemseo import sample_disciplines
from gemseo.mlearning.regression.algos.pce import PCERegressor
from gemseo.mlearning.regression.algos.pce_settings import PCERegressor_Settings
from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_1
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_2
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_3
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_12
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_13
from gemseo.problems.uncertainty.ishigami.statistics import SOBOL_23
from gemseo.problems.uncertainty.ishigami.statistics import TOTAL_SOBOL_1
from gemseo.problems.uncertainty.ishigami.statistics import TOTAL_SOBOL_2
from gemseo.problems.uncertainty.ishigami.statistics import TOTAL_SOBOL_3
from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis

from gemseo_umdo.visualizations.sobol_graph import SobolGraph

# %%
# ## Analytical Sobol' indices
#
# As the Sobol' indices of the Ishigami function are perfectly known,
# we can draw the Sobol' graph directly from these indices:
sobol_graph = SobolGraph(
    {"X1": SOBOL_1, "X2": SOBOL_2, "X3": SOBOL_3},
    {
        "X1": TOTAL_SOBOL_1,
        "X2": TOTAL_SOBOL_2,
        "X3": TOTAL_SOBOL_3,
    },
    {
        ("X1", "X2"): SOBOL_12,
        ("X1", "X3"): SOBOL_13,
        ("X2", "X3"): SOBOL_23,
    },
)
sobol_graph

# %%
# We can see that
# the thickness of a node is proportional to the corresponding total index
# (this is the first value in the node, the second one being the first-order index)
# while an edge represents a second-order index.
#
# !!! note
#     A [SobolGraph][gemseo_umdo.visualizations.sobol_graph.SobolGraph]
#     can easily be displayed in an HTML page or in a Jupyter notebook.
#     You can also use its `visualize` method
#     to save it on the disk or display it with a dedicated program.
#
# ## Indices estimated by sampling
#
# If they were not known,
# we could also estimate them by sampling.
#
# First,
# we estimate the indices:
sobol_analysis = SobolAnalysis()
sobol_analysis.compute_samples([IshigamiDiscipline()], IshigamiSpace(), 1000)
sobol_analysis.compute_indices()

# %%
# Then, we plot the Sobol' graph:
sobol_graph = SobolGraph.from_analysis(sobol_analysis, "y")
sobol_graph
# %%
# and find a figure similar to the previous one.
#
# ## Indices estimated from a PCE
#
# We could also estimate them from a polynomial chaos expansion
# (PCE, see [PCERegressor][gemseo.mlearning.regression.algos.pce.PCERegressor]).
#
# ### Create the training dataset
# First,
# we create a training dataset by sampling the Ishigami function using an optimal LHS:
samples = sample_disciplines(
    [IshigamiDiscipline()],
    IshigamiSpace(
        uniform_distribution_name=IshigamiSpace.UniformDistribution.OPENTURNS
    ),
    ["y"],
    algo_name="OT_OPT_LHS",
    n_samples=50,
)

# %%
# ### Create the PCE
# Then,
# we create a PCE using the LARS technique:
pce_settings = PCERegressor_Settings(degree=6, use_lars=True)
pce = PCERegressor(samples, settings_model=pce_settings)
pce.learn()

# %%
# ### Create the graph
# Lastly, we plot the Sobol' graph:
sobol_graph = SobolGraph.from_pce(pce, "y")
sobol_graph

# %%
# We find a figure similar to the previous ones.
