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
"""The GEMSEO-free spring-mass model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import arange
from scipy.integrate import odeint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class SpringMassModel:
    r"""The GEMSEO-free spring-mass model.

    This model computes the time displacement of an object attached to a spring in
    function of the stiffness of the spring.

    It computes also its maximum displacement.

    The ordinary differential equation is

    $$m\frac{d^2z(t)}{dt^2} = -kz(t) + mg$$

    with $\left.\frac{dz(t)}{dt}\right|_{t=0}=z(0)=z_0$.
    """

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
        self.__mass = mass
        self.__gravity = gravity
        self.__initial_state = initial_state
        self.__time = arange(initial_time, final_time, time_step)
        self.__cost = 1.0 / time_step

    @property
    def cost(self) -> float:
        """The evaluation cost."""
        return self.__cost

    def __call__(self, stiffness: float = 2.25) -> tuple[NDArray[float], float]:
        """Compute the displacement of the object w.r.t. the stiffness of the spring.

        Args:
            stiffness: The stiffness of the spring.

        Returns:
            The displacement of the object at the different times,
            as well as its maximum displacement.
        """
        displacements = odeint(
            self.__integration_func,
            self.__initial_state,
            self.__time,
            args=(stiffness, self.__mass, self.__gravity),
        )[:, 0]
        return (displacements, max(displacements))

    @staticmethod
    def __integration_func(
        state: Sequence[float], t: float, k: float, m: float, g: float
    ) -> list[float, float]:
        """Compute the derivative of the state (velocity/acceleration) at a given time.

        Args:
            state: The velocity and acceleration of the object.
            t: The time.
            k: The stiffness of the spring.
            m: The mass of the object.
            g: The gravity acceleration.

        Returns:
            The derivative of the velocity,
            the derivative of the acceleration.
        """
        return [state[1], -k * state[0] / m + g]
