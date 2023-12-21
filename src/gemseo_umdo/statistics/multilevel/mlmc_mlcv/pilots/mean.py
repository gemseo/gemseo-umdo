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
"""The mean-based pilot for the MLMC-MLCV algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from numpy import array
from numpy import cov
from numpy import dot
from numpy import nanmean
from numpy import nansum
from numpy import nanvar
from scipy.linalg import solve

from gemseo_umdo.statistics.multilevel.mlmc_mlcv.mlmc_mlcv import MLMCMLCV
from gemseo_umdo.statistics.multilevel.mlmc_mlcv.pilots.pilot import MLMCMLCVPilot

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy.typing import NDArray


class Mean(MLMCMLCVPilot):
    """The mean-based pilot for the MLMC-MLCV algorithm."""

    def __init__(  # noqa: D107
        self, sampling_ratios: NDArray[float], costs: NDArray[float]
    ) -> None:
        super().__init__(sampling_ratios, costs)
        self.__delta = [array([]) for _ in range(len(sampling_ratios))]

    def _compute_statistic(self) -> float:  # noqa: D102
        # Adapt
        # El Amri et al., Multilevel Surrogate-based Control Variates, 2023.
        # (Eq. 45, 52-57)
        # based on the linearity of the mean.
        return nansum([nanmean(delta) for delta in self.__delta])

    def _compute_V_l(  # noqa: D102 N802
        self,
        levels: Iterable[int],
        samples: Sequence[NDArray[float]],
        *pilot_parameters: Any,
    ) -> NDArray[float]:
        g_means, h_means, mlmc_mlcv_variant = pilot_parameters
        for level in levels:
            _samples = samples[level]
            f_delta = (_samples[:, 0] - _samples[:, 1]).reshape((-1, 1))  # noqa: N806
            if mlmc_mlcv_variant == MLMCMLCV.Variant.MLMC_CV_0 and level != 0:
                self.__delta[level] = f_delta.ravel()
            else:
                positions = MLMCMLCV.get_surrogate_positions(
                    level, len(self.V_l), mlmc_mlcv_variant
                )
                # In the following, "sm" stands for "surrogate model".
                sm_means = g_means[positions] if level == 0 else h_means[positions]
                sm_samples = _samples[:, 2:]  # noqa: N806
                cov_f_sm = dot(  # noqa: N806
                    f_delta.T - f_delta.mean(), sm_samples - sm_samples.mean(axis=0)
                ) / len(sm_samples)
                alpha = solve(cov(sm_samples.T, bias=True), cov_f_sm.T, assume_a="pos")
                self.__delta[level] = (
                    f_delta - (sm_samples - sm_means) @ alpha
                ).ravel()

        return array([nanvar(delta) for delta in self.__delta])
