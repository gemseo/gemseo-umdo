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
"""# Sobol' graph for the Sellar use case."""

from __future__ import annotations

from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis

from gemseo_umdo.visualizations.sobol_graph import SobolGraph

# %%
# First,
# we consider the
# [SellarDesignSpace][gemseo.problems.mdo.sellar.sellar_design_space.SellarDesignSpace]
# as the uncertain space,
# which means that the uncertain variables are the design variables
# uniformly distributed between their lower and upper bounds:
design_space = SellarDesignSpace(dtype="float64")

# %%
# Then,
# we define the disciplines:
disciplines = [Sellar1(), Sellar2(), SellarSystem()]

# %%
# Thirdly,
# we compute the Sobol' indices for all the outputs of the MDO problem:
sobol_analysis = SobolAnalysis()
sobol_analysis.compute_samples(disciplines, design_space, 100)
sobol_analysis.compute_indices()

# %%
# Lastly,
# we draw the Sobol' graph :
sobol_graph = SobolGraph.from_analysis(sobol_analysis, output_name="obj")
sobol_graph
