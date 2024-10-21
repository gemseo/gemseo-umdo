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
r"""# Robust optimization problem.

Minimize the expectation of the weight $w(h,t)$ w.r.t. the height $h\in[500, 800]$ and
the thickness $t\in[2,10]$ while satisfying $c_{\text{stress}}(h,t)\geq 1.0$ and
$c_{\text{displacement}}(h,t)\leq 1.0$ with probability 90\% where $F$, $E$ and
$\sigma_{\text{all}}$ are random variables defined by `BeamUncertainSpace`.
"""

from __future__ import annotations

from gemseo import configure
from gemseo import configure_logger

from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario
from gemseo_umdo.use_cases.beam_model.constraints import BeamConstraints
from gemseo_umdo.use_cases.beam_model.design_space import BeamDesignSpace
from gemseo_umdo.use_cases.beam_model.discipline import Beam
from gemseo_umdo.use_cases.beam_model.uncertain_space import BeamUncertainSpace

configure()
configure_logger()

scenario = UMDOScenario(
    [Beam(), BeamConstraints()],
    "MDF",
    "w",
    BeamDesignSpace(),
    BeamUncertainSpace(uniform=False),
    "Mean",
    statistic_estimation="Sampling",
    statistic_estimation_parameters={"n_samples": 200},
)
scenario.add_constraint(
    "c_stress", "Probability", greater=False, threshold=1.0, positive=True, value=0.9
)
scenario.add_constraint(
    "c_displ", "Probability", greater=True, threshold=1.0, positive=True, value=0.9
)
scenario.execute(algo_name="NLOPT_COBYLA", max_iter=30)

scenario.post_process("OptHistoryView", save=False, show=True)
