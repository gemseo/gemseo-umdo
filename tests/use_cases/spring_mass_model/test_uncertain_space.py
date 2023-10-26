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
"""Tests for SpringMassUncertainSpace."""
from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.composed import OTComposedDistribution
from gemseo_umdo.use_cases.spring_mass_model.uncertain_space import (
    SpringMassUncertainSpace,
)


def test_uncertain_space():
    """Check the content of the uncertain space used in the spring-mass use case."""
    uncertain_space = SpringMassUncertainSpace()
    assert len(uncertain_space) == 1
    assert "stiffness" in uncertain_space.uncertain_variables
    distribution = uncertain_space.distributions["stiffness"]
    assert isinstance(distribution, OTComposedDistribution)
    assert len(distribution.marginals) == 1
    distribution = uncertain_space.distributions["stiffness"]
    assert repr(distribution) == "Beta(3.0, 2.0, 1.0, 3.5)"
