# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo_umdo.scenarios.udoe_scenario import UDOEScenario


def test_scenario():
    """Check SequentialSampling."""
    discipline = AnalyticDiscipline({"y": "(x+u)**2"}, name="quadratic_function")
    discipline.set_cache_policy("MemoryFullCache")
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=-1, u_b=1.0, value=0.5)
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("u", "OTNormalDistribution")
    scenario = UDOEScenario(
        [discipline],
        "DisciplinaryOpt",
        "y",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation="SequentialSampling",
        statistic_estimation_parameters={
            "n_samples": 7,
            "initial_n_samples": 2,
            "n_samples_increment": 2,
        },
    )
    scenario.execute({"algo": "fullfact", "n_samples": 5})
    assert discipline.n_calls == (2 + 4 + 6 + 7 + 7)
