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
"""# The uncertain coupling graph for the Sobieski's SSBJ use case"""
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

from gemseo_umdo.visualizations.uncertain_coupling_graph import UncertainCouplingGraph

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
# we instantiate an uncertain coupling graph:
uncertain_coupling_graph = UncertainCouplingGraph(disciplines, uncertain_space)

# %%
# and sample the multidisciplinary system with 100 evaluations:
uncertain_coupling_graph.sample(100)

# %%
# Lastly,
# we visualize it:
uncertain_coupling_graph.visualize(save=False, show=True)

# %%
# In this example designed for Sphinx Gallery and Jupyter Notebook,
# we do not save the graph on the disk (``save=False``)
# or display it with a dedicated program (``save=True``)
# but display it in the web browser.
