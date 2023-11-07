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
"""Tests for the MLMCPilotFactory."""
from __future__ import annotations

import pytest

from gemseo_umdo.statistics.multilevel.mlmc.pilots.factory import MLMCPilotFactory


@pytest.mark.parametrize("pilot_class_name", ["Mean", "Variance"])
def test_is_available(pilot_class_name: str):
    """Check if Mean and Variance are available in the MLMCPilotFactory."""
    assert MLMCPilotFactory().is_available(pilot_class_name)
