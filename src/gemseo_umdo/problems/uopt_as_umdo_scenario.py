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
"""Make a monodisciplinary optimization problem under uncertainty multidisciplinary."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from gemseo.problems.mdo.opt_as_mdo_scenario import BaseLinkDiscipline
from gemseo.problems.mdo.opt_as_mdo_scenario import LinearLinkDiscipline
from gemseo.problems.mdo.opt_as_mdo_scenario import create_disciplines
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping

    from gemseo_umdo.formulations.base_umdo_formulation_settings import (
        BaseUMDOFormulationSettings,
    )


class UOptAsUMDOScenario(UMDOScenario):
    """An optimization scenario under uncertainty made multidisciplinary."""

    def __init__(
        self,
        discipline: Discipline,
        objective_name: str,
        design_space: DesignSpace,
        uncertain_space: ParameterSpace,
        objective_statistic_name: str,
        statistic_estimation_settings: BaseUMDOFormulationSettings,
        objective_statistic_parameters: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        uncertain_design_variables: Mapping[
            str, str | tuple[str, str]
        ] = READ_ONLY_EMPTY_DICT,
        name: str = "",
        formulation_settings_model: BaseFormulationSettings | None = None,
        maximize_objective: bool = False,
        coupling_equations: tuple[
            Iterable[Discipline, ...],
            Callable[[RealArray], RealArray],
            Callable[[RealArray], RealArray],
        ] = (),
        link_discipline_class: type[BaseLinkDiscipline] = LinearLinkDiscipline,
        **formulation_settings: Any,
    ) -> None:
        r"""
        Args:
            discipline: The discipline
                computing the objective, constraints and observables
                from the design variables.
            design_space: The design space
                including the design variables $z_0,z_1,\ldots,z_N$
                which will be replaced by $x_0,x_1,\ldots,x_N$ respectively
                in the U-MDO problem.
            coupling_equations: The objects
                to evaluate and solve the coupling equations,
                namely the disciplines $h_1,\ldots,h_N$,
                the function $c$
                and the Jacobian function $\nabla c(x)$.
                If empty,
                the $i$-th discipline is linear.
            link_discipline_class: The class of the link discipline.

        Note:
            There is no naming convention
            for the input and output variables of ``discipline``.
            So,
            you can use $a,b,c$ in ``design_space`` instead of $z_0,z_1,z_2$.
        """  # noqa: D205, D212, E501
        disciplines = create_disciplines(
            discipline, design_space, coupling_equations, link_discipline_class
        )
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            uncertain_space,
            objective_statistic_name,
            statistic_estimation_settings=statistic_estimation_settings,
            objective_statistic_parameters=objective_statistic_parameters,
            uncertain_design_variables=uncertain_design_variables,
            name=name,
            maximize_objective=maximize_objective,
            formulation_settings_model=formulation_settings_model,
            **formulation_settings,
        )
