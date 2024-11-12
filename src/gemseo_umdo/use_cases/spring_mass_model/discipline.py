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
"""The spring-mass model use case."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.core.discipline.discipline import Discipline
from numpy import array

from gemseo_umdo.use_cases.spring_mass_model.model import SpringMassModel

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


class SpringMassDiscipline(Discipline):
    r"""The GEMSEO-based spring-mass model $m\frac{d^2z(t)}{dt^2} = -kz(t) + mg$.

    This model computes the time displacement of an object attached to a spring in
    function of the stiffness of the spring.

    It computes also its maximum displacement.
    """

    __STIFFNESS: Final[str] = "stiffness"
    __DISPLACEMENT: Final[str] = "displacement"
    __MAX_DISPLACEMENT: Final[str] = "max_displacement"

    def __init__(
        self,
        mass: float = 1.5,
        initial_state: tuple[float, float] = (0, 0),
        initial_time: float = 0.0,
        final_time: float = 10.0,
        time_step: float = 0.1,
        gravity: float = 9.8,
    ) -> None:
        """
        Args:
            mass: The mass of the object.
            initial_state: The initial position and velocity of the object.
            initial_time: The initial time.
            final_time: The final time.
            time_step: The time step.
            gravity: The gravity acceleration.
        """  # noqa: D205 D212 D415
        super().__init__(name=f"{self.__class__.__name__}({time_step})")
        self.input_grammar.update_from_names([self.__STIFFNESS])
        self.output_grammar.update_from_names([
            self.__MAX_DISPLACEMENT,
            self.__DISPLACEMENT,
        ])
        self.__model = SpringMassModel(
            mass=mass,
            initial_state=initial_state,
            initial_time=initial_time,
            final_time=final_time,
            time_step=time_step,
            gravity=gravity,
        )
        self.default_input_data = {self.__STIFFNESS: array([2.25])}

    def _run(self, input_data: StrKeyMapping) -> None:
        disp, max_disp = self.__model(self.io.data[self.__STIFFNESS][0])
        self.io.data[self.__DISPLACEMENT] = disp
        self.io.data[self.__MAX_DISPLACEMENT] = array([max_disp])

    @property
    def cost(self) -> float:
        """The evaluation cost."""
        return self.__model.cost
