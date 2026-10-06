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
"""Make a monodisciplinary optimization problem under uncertainty multidisciplinary.

GEMSEO proposes features
to make a monodisciplinary optimization problem multidisciplinary.
Here is an extension to MDO under uncertainty.
Please read the
[GEMSEO's documentation](https://gemseo.readthedocs.io/en/stable/modules/gemseo.problems.mdo.opt_as_mdo_scenario.html)
to get more information about the basics of this technique.

!!! quote "References"
    Aziz Alaoui,
    Contributions to multidisciplinary design optimization under uncertainty,
    with applications to aircraft design.
    General Mathematics [math.GM]. Université de Toulouse, 2025.
    English.
    [⟨NNT : 2025TLSEI001⟩](https://www.theses.fr/2025TLSEI001).
    [⟨tel-05059696⟩](https://theses.hal.science/tel-05059696v1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.problem.mdo.opt_as_mdo_scenario import LinearLinkDiscipline
from gemseo.problem.mdo.opt_as_mdo_scenario import create_disciplines
from gemseo.util.constant import read_only_empty_dict

from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.space import DesignSpace
    from gemseo.space import RandomSpace
    from gemseo.discipline import Discipline
    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.problem.mdo.opt_as_mdo_scenario import BaseLinkDiscipline
    from gemseo.util.typing import RealArray
    from gemseo.util.typing import StrKeyMapping

    from gemseo_umdo.formulations.base_umdo_formulation_settings import (
        BaseUMDOFormulationSettings,
    )


class UOptAsUMDOScenario(UMDOScenario):
    """An optimization scenario under uncertainty made multidisciplinary."""

    # TODO(bump-gemseo): rename would duplicate an existing parameter, not renamed: formulation_settings_model -> formulation_settings  # noqa: E501
    def __init__(
        self,
        discipline: Discipline,
        objective_name: str,
        design_space: DesignSpace,
        uncertain_space: RandomSpace,
        objective_statistic_name: str,
        statistic_estimation_settings: BaseUMDOFormulationSettings,
        objective_statistic_parameters: StrKeyMapping = read_only_empty_dict,
        uncertain_design_variables: Mapping[
            str, str | tuple[str, str]
        ] = read_only_empty_dict,
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
        # TODO(bump-gemseo): cannot transform: objective_name goes to a call of add_objective() on the result of this call, which is not assigned to a name by a statement of its own  # noqa: E501
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
