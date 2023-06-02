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
Sobol' graph for the Sobieski's SSBJ use case
=============================================
"""
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from gemseo.uncertainty.sensitivity.sobol.analysis import SobolAnalysis
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo_umdo.visualizations.sobol_graph import SobolGraph

# %%
# First,
# we define an uncertain space around the optimum design:
design_space = SobieskiProblem().design_space
design_variable_names = ["x_1", "x_2", "x_3", "x_shared"]
design_space.filter(design_variable_names)
optimum_design = split_array_to_dict_of_arrays(
    SobieskiProblem().optimum_design,
    design_space.variable_sizes,
    design_variable_names,
)

uncertain_space = DesignSpace()
for name, value in optimum_design.items():
    uncertain_space.add_variable(
        name,
        size=value.size,
        l_b=value * 0.95,
        u_b=value * 1.05,
        value=value,
    )

# %%
# Then,
# we define the disciplines:
disciplines = [
    SobieskiAerodynamics(),
    SobieskiStructure(),
    SobieskiPropulsion(),
    SobieskiMission(),
]

# %%
# Thirdly,
# we compute the Sobol' indices for all the outputs of the MDO problem:
sobol_analysis = SobolAnalysis(disciplines, uncertain_space, 100)
sobol_analysis.compute_indices()

# %%
# Lastly,
# we draw the Sobol' graph:
sobol_graph = SobolGraph.from_analysis(sobol_analysis, output_name="y_4")
sobol_graph.visualize(show=True)
