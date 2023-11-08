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
"""The GEMSEO-free version of the model for the beam use case."""

from __future__ import annotations

import numpy as np
from numpy import hstack
from numpy import linspace
from numpy import meshgrid
from numpy import sqrt

from gemseo_umdo.use_cases.beam_model.core.output_data import BeamModelOutputData
from gemseo_umdo.use_cases.beam_model.core.variables import E
from gemseo_umdo.use_cases.beam_model.core.variables import F
from gemseo_umdo.use_cases.beam_model.core.variables import L
from gemseo_umdo.use_cases.beam_model.core.variables import alpha
from gemseo_umdo.use_cases.beam_model.core.variables import b
from gemseo_umdo.use_cases.beam_model.core.variables import beta
from gemseo_umdo.use_cases.beam_model.core.variables import dy
from gemseo_umdo.use_cases.beam_model.core.variables import dz
from gemseo_umdo.use_cases.beam_model.core.variables import h
from gemseo_umdo.use_cases.beam_model.core.variables import nu
from gemseo_umdo.use_cases.beam_model.core.variables import rho
from gemseo_umdo.use_cases.beam_model.core.variables import t


class BeamModel:
    r"""The beam model.

    We consider an horizontal beam
    with length $L$, width $b$ and height $h$.
    This beam is hollow and made of a material
    with a Young's modulus $E$,
    a Poisson's ratio $\nu$
    and a thickness $t$.
    One of its ends is fixed at $x=0$ (the "root")
    while the other  at $x=L$ (the "tip") is free.
    The $y$-axis is horizontal and perpendicular to the beam,
    the $z$ is vertical
    and the center of the root is at the origin $(0, 0, 0)$.

    A force $\vec{F}$ of amplitude $F$ is applied to the beam
    at $(L, dy, dz)$
    with an angle $\alpha$ w.r.t. $-\vec{e}_y$ in the xz-plane
    and an angle $\beta$ w.r.t. $-\vec{e}_y$ in the yz-plane,
    where $\vec{e}_y$ is the unit vector along the $y$-axis.

    From these settings,
    the model computes the weight of the beam $w=2 \rho L (b + h -2t)$
    and several quantities on a regular $yz$-grid:

    - the strain energy vector $\vec{U}=(U_x,U_y,U_z)$ at the tip,
    - the normal stress $\sigma$ at the root,
    - the torsional stress $\tau$ at the root,
    - the displacement $\delta$ at the tip,
    - the von Mises stress $\sigma_{\text{VM}}$ at the root.

    The equations are:

    - Force components
        - $F_x=F\sin(\alpha)$
        - $F_y=F\cos(\alpha)\sin(\beta)$
        - $F_z=F\cos(\alpha)\cos(\beta)$
    - Inertia
        - $I_x=(2tb^2h^2)/(b + h)$
        - $I_y=(bh^3-(b-2t)(h-2t)^3)/12$
        - $I_z=(hb^3-(h-2t)(b-2t)^3)/12$
    - Strain energy
        - $U_x = E^{-1} \{ \frac{ F_x L }{ 2t (b+h-2t) } +
          zL (F_x dZ - F_z L/2) I_y^{-1} - yL (F_y L/2 - F_x dY) I_z^{-1} \}$
        - $U_y = E^{-1} \{ (F_y L^3/3 - F_x dY L^2/2)I_z^{-1} -
          zL \frac{ F_zdY-F_ydZ }{ 2 (1+\nu) I_x } \}$
        - $U_z = E^{-1} \{ (F_z L^3/3 - F_xdZ L^2/2) I_y^{-1} +
          yL \frac{ F_zdY-F_ydZ }{ 2 (1+\nu) I_x } \}$
    - Displacements
        - $\delta=\sqrt{U_x^2+U_y^2+U_z^2}$
    - Torsional stress
        - $\tau_x=(F_zdY-F_ydZ)/(2bht)$
        - $\tau_y= - (0.5|z|(b-t)+(b-t)^2(1-4y^2/(b-t)^2))F_y\text{sign}(z)/(8I_z)$
        - $\tau_z=(0.5|y|(h-t)+(h-t)^2(1-4z^2/(h-t)^2))F_z\text{sign}(y)/(8I_y)$
        - $\tau = \tau_x + \tau_y + \tau_z$
    - Stress
        - $\sigma = F_x/(2t(b+h-2t)) + y (F_xdY-F_yL)/I_z + z (F_xdZ-F_zL)/I_y$
    - von Mises stress
        - $\sigma_{\text{VM}} = \sqrt{\sigma^2 + 3\tau^2}$
    """

    def __init__(self, n_y: int = 3, n_z: int = 3) -> None:
        """
        Args:
            n_y: The number of discretization points in the y-direction.
            n_z: The number of discretization points in the z-direction.
        """  # noqa: D205 D212 D415
        self.__n_y = n_y
        self.__n_z = n_z

    def __call__(
        self,
        b: float = b.value,
        h: float = h.value,
        t: float = t.value,
        L: float = L.value,  # noqa: N803
        E: float = E.value,  # noqa: N803
        alpha: float = alpha.value,
        beta: float = beta.value,
        dy: float = dy.value,
        dz: float = dz.value,
        rho: float = rho.value,
        F: float = F.value,  # noqa: N803
        nu: float = nu.value,
    ) -> BeamModelOutputData:
        r"""Compute the weight of the beam as well as properties on a yz-grid.

        Args:
            b: The width of the beam.
            h: The height of the beam.
            t: The thickness of the beam.
            L: The length of the beam.
            E: The Young's modulus of the material.
            alpha: The angle between $-\vec{e}_z$ and $\vec{F}$
                in the $xz$-plane.
            beta: The angle between $-\vec{e}_z$ and $\vec{F}$
                in the $yz$-plane.
            dy: The $y$-coordinate of the point where the force is applied.
            dz: the $z$-coordinate of the point where the force is applied
            rho: The density of the material.
            F: The load applied to a point at the tip of the beam.
            nu: The Poisson's ratio.

        Returns:
            The strain energy along the $x$-axis,
            the strain energy along the $y$-axis,
            the strain energy along the $z$-axis,
            the normal stress at the root section points,
            the torsional stress at the root section points,
            the displacement at the tip section points,
            the von Mises stress at the root section points
            and the weight of the beam.
        """
        y, z = meshgrid(
            linspace(-b / 2, b / 2, self.__n_y), linspace(-h / 2, h / 2, self.__n_z)
        )

        # Compute the inertia vector.
        I_x = 2 * (b * h) ** 2 * t / (b + h)  # noqa: N806
        I_y = b * h**3 / 12 - (b - 2 * t) * (h - 2 * t) ** 3 / 12  # noqa: N806
        I_z = h * b**3 / 12 - (h - 2 * t) * (b - 2 * t) ** 3 / 12  # noqa: N806

        # Compute the force vector.
        F_x = F * np.sin(alpha)  # noqa: N806
        F_y = F * np.cos(alpha) * np.sin(beta)  # noqa: N806
        F_z = F * np.cos(alpha) * np.cos(beta)  # noqa: N806

        # Compute the tip moments.
        M_tip_y = F_x * dz  # noqa: N806
        M_tip_z = -F_x * dy  # noqa: N806

        # Compute the strain energy.
        U_x = F_x / 2 / t / (b + h - 2 * t) / E * L  # noqa: N806
        U_y = F_y * L**3 / 3 / E / I_z + M_tip_z * L**2 / 2 / E / I_z  # noqa: N806
        U_z = F_z * L**3 / 3 / E / I_y - M_tip_y * L**2 / 2 / E / I_y  # noqa: N806

        # Compute the moments.
        M_x = F_z * dy - F_y * dz  # noqa: N806
        M_y = M_tip_y - F_z * L  # noqa: N806
        M_z = M_tip_z + F_y * L  # noqa: N806

        # Compute theta.
        theta_x = M_x / (E / 2 / (1 + nu)) / I_x * L
        theta_y = -F_z * L**2 / 2 / E / I_y + M_tip_y * L / E / I_y
        theta_z = F_y * L**2 / 2 / E / I_z + M_tip_z * L / E / I_z

        # Compute the stress.
        sigma_xx = F_x / 2 / t / (b + h - 2 * t)
        sigma_xy = -M_z / I_z * y
        sigma_xz = M_y / I_y * z

        # Compute S_t.
        S_t_y = abs(y) * (h / 2 - t / 2) + (h - t) ** 2 / 8 * (  # noqa: N806
            1 - 4 * z**2 / (h - t) ** 2
        )
        S_t_z = abs(z) * (b / 2 - t / 2) + (b - t) ** 2 / 8 * (  # noqa: N806
            1 - 4 * y**2 / (b - t) ** 2
        )

        # Compute the torsional stress.
        tau_xx = M_x / 2 / b / h / t
        tau_xy = -F_y / I_z * S_t_z * np.sign(z)
        tau_xz = F_z / I_y * S_t_y * np.sign(y)

        U_z = U_z + y * theta_x  # noqa: N806
        U_y = U_y - z * theta_x  # noqa: N806
        U_x = U_x + z * theta_y - y * theta_z  # noqa: N806
        sigma = sigma_xz + sigma_xy + sigma_xx
        tau = tau_xz + tau_xy + tau_xx
        return BeamModelOutputData(
            U_x,
            U_y,
            U_z,
            sigma,
            tau,
            sqrt(U_x**2 + U_y**2 + U_z**2),
            sqrt(sigma**2 + 3 * tau**2),
            2.0 * rho * L * (b + h - 2.0 * t) * t,
            hstack((y.reshape((-1, 1)), z.reshape((-1, 1)))),
        )
