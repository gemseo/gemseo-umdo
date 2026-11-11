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
"""Tools for the estimation of statistics.

This package exposes the settings of the statistics
to be estimated by the U-MDO formulations,
e.g. [Margin_Settings][gemseo_umdo.statistic.settings.margin.Margin_Settings].
"""

from __future__ import annotations

from gemseo_umdo.statistic.settings.margin import Margin_Settings
from gemseo_umdo.statistic.settings.mean import Mean_Settings
from gemseo_umdo.statistic.settings.probability import Probability_Settings
from gemseo_umdo.statistic.settings.quantile import Quantile_Settings
from gemseo_umdo.statistic.settings.standard_deviation import StandardDeviation_Settings
from gemseo_umdo.statistic.settings.variance import Variance_Settings

__all__ = [
    "Margin_Settings",
    "Mean_Settings",
    "Probability_Settings",
    "Quantile_Settings",
    "StandardDeviation_Settings",
    "Variance_Settings",
]
