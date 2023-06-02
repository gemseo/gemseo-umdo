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
"""Tests related to the dataclass BeamModelOutputData."""
from __future__ import annotations

from dataclasses import fields

from gemseo_umdo.use_cases.beam_model.core.output_data import (
    BeamModelOutputData,
)


def test_field_names():
    """Check the names of the fields."""
    assert ["Ux", "Uy", "Uz", "sigma", "tau", "displ", "sigma_vm", "w", "yz_grid"] == [
        f.name for f in fields(BeamModelOutputData)
    ]
