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
"""Tests for the BeamDesignSpace."""
from __future__ import annotations

from gemseo_umdo.use_cases.beam_model.core.design_space import (
    BeamDesignVariables,
)
from gemseo_umdo.use_cases.beam_model.core.design_space import DesignVariable


def test_number():
    """Check the number of design variables."""
    assert len(BeamDesignVariables) == 2


def test_h():
    """Check the value of the design variable h."""
    assert BeamDesignVariables.h.value == DesignVariable("h", 500, 800, 800)


def test_t():
    """Check the value of the design variable t."""
    assert BeamDesignVariables.t.value == DesignVariable("t", 2, 10, 2.5)
