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
r"""# Large DOE study

Sample the weight $w(h,t)$ and the constraints $c_{\text{stress}}(h,t)$ and
$c_{\text{displacement}}(h,t)$ w.r.t. the height $h\in[500, 800]$ and the thickness
$t\in[2,10]$.
"""
from __future__ import annotations

from gemseo import configure_logger
from gemseo.core.doe_scenario import DOEScenario
from gemseo.post.dataset.zvsxy import ZvsXY

from gemseo_umdo.use_cases.beam_model.constraints import BeamConstraints
from gemseo_umdo.use_cases.beam_model.design_space import BeamDesignSpace
from gemseo_umdo.use_cases.beam_model.discipline import Beam

configure_logger()

disciplines = [Beam(), BeamConstraints()]

design_space = BeamDesignSpace()

scenario = DOEScenario(disciplines, "MDF", "w", design_space)
scenario.add_constraint("c_stress", constraint_type="ineq", positive=True, value=1.0)
scenario.add_constraint("c_displ", constraint_type="ineq", value=1.0)
scenario.execute({"algo": "fullfact", "n_samples": 10**2})

dataset = scenario.formulation.opt_problem.to_dataset()
ZvsXY(dataset, "h", "t", "w").execute(save=True, show=False, file_name="w")
for constraint_name in ["[c_displ-1.0]", "-[c_stress-1.0]"]:
    for z_component in range(9):
        ZvsXY(dataset, "h", "t", (constraint_name, z_component)).execute(
            save=False,
            show=True,
        )
