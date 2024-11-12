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
"""Multilevel Monte Carlo with multilevel control variates (MLMC-MLCV)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.utils.seeder import SEED
from numpy import array
from strenum import StrEnum

from gemseo_umdo.statistics.multilevel.mlmc.mlmc import MLMC
from gemseo_umdo.statistics.multilevel.mlmc_mlcv.pilots.factory import (
    MLMCMLCVPilotFactory,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.mdo_functions.mdo_function import MDOFunction

    from gemseo_umdo.statistics.multilevel.mlmc_mlcv.level import Level


class MLMCMLCV(MLMC):
    """Multilevel Monte Carlo with multilevel control variates (MLMC-MLCV)."""

    class Variant(StrEnum):
        """A variant of the MLMC-MLCV algorithm."""

        MLMC_MLCV = "MLMC-MLCV"
        MLMC_MLCV_0 = "MLMC-MLCV[0]"
        MLMC_CV = "MLMC-CV"
        MLMC_CV_0 = "MLMC-CV[0]"

    _PILOT_FACTORY = MLMCMLCVPilotFactory

    __g_l: tuple[MDOFunction]
    r"""The control variates $g_0$, $g_1$, ..., $g_L$.

    $g_\ell$ is an approximation of $f_\ell$.
    """

    __h_l: tuple[MDOFunction]
    r"""The control variates $h_1$, ..., $h_L$.

    $h_\ell$ is an approximation of $f_\ell-f_{\ell-1}$.
    """

    __variant: Variant
    """A variant of the algorithm."""

    def __init__(  # noqa: D107
        self,
        levels: Sequence[Level],
        uncertain_space: ParameterSpace,
        n_samples: float,
        pilot: str = "Mean",
        variant: Variant = Variant.MLMC_MLCV,
        seed: int = SEED,
    ) -> None:
        self.__g_l = tuple(level.surrogate_model[0] for level in levels)
        for l, g_l in enumerate(self.__g_l):  # noqa: E741
            g_l.name = f"g[{l}]"

        self.__h_l = tuple(
            level.difference_surrogate_model[0]
            for l, level in enumerate(levels)  # noqa: E741
            if l != 0
        )
        for l, h_l in enumerate(self.__h_l):  # noqa: E741
            h_l.name = f"h[{l + 1}]"
        self.__variant = variant
        super().__init__(
            levels,
            uncertain_space,
            n_samples,
            pilot_statistic_name=pilot,
            seed=seed,
        )

        self._algorithm_name = variant.value
        self._pilot_statistic_estimator_parameters = [
            array([level.surrogate_model[1] for level in levels]),
            array([
                level.difference_surrogate_model[1]
                for l, level in enumerate(levels)  # noqa: E741
                if l != 0
            ]),
            variant,
        ]

    def _add_functions_to_samplers(self) -> None:
        # At level l, sample the models f[l] and f[l-1].
        super()._add_functions_to_samplers()
        # At level l, sample some surrogate models.
        for l, sampler in enumerate(self._samplers):  # noqa: E741
            positions = self.get_surrogate_positions(l, self._n_levels, self.__variant)
            surrogate_models = self.__h_l[positions] if l else self.__g_l[positions]
            for surrogate_model in surrogate_models:
                sampler.add_function(surrogate_model)

    @classmethod
    def get_surrogate_positions(
        cls, level: int, n_levels: int, variant: Variant
    ) -> slice:
        r"""Return the positions of the surrogate models for given level and variant.

        These are their positions in a sequence starting to count at 0.
        So,
        the position of $g_\ell$ is $\ell$
        for $\ell\in\{0,\ldots,L\}$
        while
        the position of $h_\ell$ is $\ell-1$
        for $\ell\in\{1,\ldots,L\}$.

        Args:
            level: The level of the telescopic sum.
            n_levels: The number of levels.
            variant: The variant of the algorithm.

        Returns:
            The positions of the surrogate models.

        See Also:
            El Amri et al., Table 1, Multilevel Surrogate-based Control Variates, 2023.
        """
        if level == 0:
            if variant == cls.Variant.MLMC_MLCV:
                return slice(0, n_levels)

            if variant == cls.Variant.MLMC_CV:
                return slice(0, 1)

            if variant == cls.Variant.MLMC_MLCV_0:
                return slice(0, 2)

            return slice(0, 1)

        if variant == cls.Variant.MLMC_MLCV:
            return slice(0, n_levels - 1)

        if variant == cls.Variant.MLMC_CV:
            return slice(level - 1, level)

        if variant == cls.Variant.MLMC_MLCV_0:
            return slice(0, 1)

        return slice(0, 0)
