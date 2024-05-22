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
r"""# Sobol' sensitivity analysis.

Compute and plot the total Sobol' indices for the field constraints $c_{\text{stress}}$
and $c_{\text{displacement}}$ where $F$, $E$ and $\sigma_{\text{all}}$ are random
variables defined by `BeamUncertainSpace`.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo.core.chain import MDOChain
from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis

from gemseo_umdo.use_cases.beam_model.constraints import BeamConstraints
from gemseo_umdo.use_cases.beam_model.discipline import Beam
from gemseo_umdo.use_cases.beam_model.uncertain_space import BeamUncertainSpace

configure_logger()

uncertain_space = BeamUncertainSpace()

n_y = n_z = 10

mdo_chain = MDOChain([Beam(n_y=n_y, n_z=n_z), BeamConstraints()])

sobol = SobolAnalysis(
    [mdo_chain], uncertain_space, 500, output_names=["c_displ", "c_stress"]
)
mesh = mdo_chain.disciplines[0].local_data["yz_grid"].reshape((-1, 2))
sobol.main_method = "total"
sobol.compute_indices()
sobol.plot_field("c_displ", mesh=mesh, save=False, show=True)
sobol.plot_field("c_stress", mesh=mesh, save=False, show=True)
