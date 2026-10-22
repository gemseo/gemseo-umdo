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
"""Scenario for evaluating statistics of disciplinary outputs under uncertainty."""

from __future__ import annotations

from gemseo.scenario import EvaluationScenario
from gemseo.space import DesignSpace

from gemseo_umdo.scenarios.base_u_scenario import BaseUScenario


class UEvaluationScenario(BaseUScenario, EvaluationScenario[DesignSpace]):
    """A scenario for evaluating statistics of disciplinary outputs under uncertainty.

    The statistics of the outputs of interest are declared as observables
    using the
    [add_observable()][gemseo_umdo.scenarios.base_u_scenario.BaseUScenario.add_observable]
    method
    and evaluated over the design space with a DOE algorithm.

    Use [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
    to define an objective and constraints.
    """
