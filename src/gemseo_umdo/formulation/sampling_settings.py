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

from pathlib import Path

from pydantic import Field

from gemseo_umdo.formulation.base_sampling_settings import BaseSamplingSettings


class Sampling_Settings(BaseSamplingSettings):  # noqa: N801
    """The settings for the sampling-based U-MDO formulation.

    The estimation of a variance, a standard deviation
    or a margin
    requires at least 2 samples of the uncertain variables.
    """

    estimate_statistics_iteratively: bool = Field(
        default=True,
        description="""Whether to estimate the statistics iteratively.

This can be useful for memory reasons.

When the evaluation raises an error at a sample of the uncertain variables,
e.g. that of an output or of a gradient,
the DOE algorithm skips this sample.
In this case,
the iterative estimation does not use this sample for any output,
while the non-iterative one uses the outputs evaluated before the error
(and, to estimate the Jacobians, their gradients),
as they are stored in the database.
So, the statistics of an output can be estimated from fewer samples
with the iterative estimation;
in particular,
when the gradient of an output fails at all the samples,
the iterative estimation reports that the output is not available
while the non-iterative one reports that its gradient is not available.

This argument is ignored when `samples_directory_path` is defined;
in this case, the statistics are not estimated iteratively.

Some statistics, e.g. the quantile, cannot be estimated iteratively;
using them requires setting this argument to `False`.""",
    )

    samples_directory_path: str | Path = Field(
        default="",
        description="""The path to a new directory
where the samples stored as `IODataset` objects will be saved
(one object per file, one file per iteration).
This directory must not exist; it will be created by the formulation.
An object stores all the evaluated samples,
including those whose evaluation failed after some outputs were computed,
which are used to estimate the statistics of an output
only if this output and, when required, its gradient were computed at them,
and stores only once a sample generated several times by the DOE algorithm.
If empty, do not save the samples.""",
    )
