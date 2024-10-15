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
"""A noising discipline."""

from __future__ import annotations

from typing import ClassVar

from gemseo.core.discipline.discipline import Discipline


class BaseNoiser(Discipline):
    """A discipline noising a variable.

    [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
    and
    [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
    create this kind of discipline
    when using their argument `uncertain_design_variables`
    in order to define the link between design and uncertain variables
    in an intuitive way.
    """

    _noised_variable_name: str
    """The name of the variable once noised."""

    _variable_name: str
    """The name of the variable to be noised."""

    _uncertain_variable_name: str
    """The name of the uncertain variable."""

    SHORT_NAME: ClassVar[str] = ""
    """A short name of the noising discipline to instantiate it more easily.

    For example,
    `"*"` would be a good short name for a `"MultiplicativeNoiser"`,
    *i.e.* clear and concise.

    In particular,
    this short name can be used to set the `uncertain_design_variables` argument
    of
    [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
    and
    [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario].
    """

    def __init__(
        self,
        variable_name: str,
        noised_variable_name: str,
        uncertain_variable_name: str,
    ) -> None:
        """
        Args:
            variable_name: The name of the variable to be noised.
            noised_variable_name: The name of the variable once noised.
            uncertain_variable_name: The name of the uncertain variable.
        """  # noqa: D205 D212 D415
        self._noised_variable_name = noised_variable_name
        self._variable_name = variable_name
        self._uncertain_variable_name = uncertain_variable_name
        super().__init__()
        self.input_grammar.update_from_names([variable_name, uncertain_variable_name])
        self.output_grammar.update_from_names([noised_variable_name])
