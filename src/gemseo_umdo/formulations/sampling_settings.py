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
"""Settings for the sampling-based U-MDO formulation."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from pydantic import Field

from gemseo_umdo.formulations.base_sampling_settings import BaseSamplingSettings


class Sampling_Settings(BaseSamplingSettings):  # noqa: N801
    """The settings for the sampling-based U-MDO formulation."""

    _TARGET_CLASS_NAME = "Sampling"

    estimate_statistics_iteratively: bool = Field(
        default=True,
        description="""Whether to estimate the statistics iteratively.

This can be useful for memory reasons.

This argument is ignored when `samples_directory_path` is defined;
in this case, the statistics are not estimated iteratively.""",
    )

    samples_directory_path: str | Path = Field(
        default="",
        description="""The path to a new directory
where the samples stored as `IODataset` objects will be saved
(one object per file, one file per iteration).
This directory must not exist; it will be created by the formulation.
If empty, do not save the samples.""",
    )
