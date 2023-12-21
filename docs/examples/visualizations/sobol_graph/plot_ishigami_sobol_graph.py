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
"""# Sobol' graph for the Ishigami use case"""

from __future__ import annotations

from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_1
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_2
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_3
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_12
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_13
from gemseo.uncertainty.use_cases.ishigami.statistics import SOBOL_23
from gemseo.uncertainty.use_cases.ishigami.statistics import TOTAL_SOBOL_1
from gemseo.uncertainty.use_cases.ishigami.statistics import TOTAL_SOBOL_2
from gemseo.uncertainty.use_cases.ishigami.statistics import TOTAL_SOBOL_3

from gemseo_umdo.visualizations.sobol_graph import SobolGraph

# %%
# First,
# we define the first-, second- and total-order Sobol' indices
# of the different uncertain variables:
first_order_indices = {"X1": SOBOL_1, "X2": SOBOL_2, "X3": SOBOL_3}
total_order_indices = {"X1": TOTAL_SOBOL_1, "X2": TOTAL_SOBOL_2, "X3": TOTAL_SOBOL_3}
second_order_indices = {
    ("X1", "X2"): SOBOL_12,
    ("X1", "X3"): SOBOL_13,
    ("X2", "X3"): SOBOL_23,
}

# %%
# Then,
# we draw the Sobol' graph:
sobol_graph = SobolGraph(
    first_order_indices,
    second_order_indices=second_order_indices,
    total_order_indices=total_order_indices,
)
sobol_graph

# %%
# Sphinx Gallery and Jupyter Notebook can display ``sobol_graph`` in the web browser.
# You can also use ``sobol_graph.visualize()``
# to save it on the disk
# or display it with a dedicated program.
