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
r"""The heat equation model.

This model solves the 1D transient equation, a.k.a. heat equation.
It describes the temperature evolution $u$ in a $L$-length rod
from the initial time 0 to the final time $T$
with a thermal diffusivity $\nu(\mathbf{X})$
depending on a random vector $\mathbf{X}$.
The heat equation is

$$\frac{\partial u(x,t;\mathbf{X})}{\partial t}
   - \nu(\mathbf{X})\frac{\partial^2 u(x,t;\mathbf{X})}{\partial x^2} = 0$$

with the boundary condition $u(0,t;\mathbf{X})=u(L,t;\mathbf{X})=0$
where $x\in\mathcal{D}=[0,L]$ and $t\in[0,T]$.

To obtain an analytical solution,
Geraci et al. (2015) chose $L=1$ and the uncertain initial condition:

$$u(x,0;\mathbf{X}) =
   \mathcal{G}(\mathbf{X})\mathcal{F}_1(x)
   +\mathcal{I}(\mathbf{X})\mathcal{F}_2(x)
$$

where

- $\mathcal{F}_1(x)=\sin(\pi x)$,
- $\mathcal{F}_2(x)=\sin(2\pi x)+\sin(3\pi x)
          +50\left(\sin(9\pi x)+\sin(21\pi x)\right)$,
- $\mathcal{I}(\mathbf{X})=3.5
         \left(\sin(X_1)+7\sin^2(X_2)+0.1X_3^4\sin(X_1)\right)$,
- $\mathcal{G}(\mathbf{X})=50\prod_{i=5}^7(4|X_i|-1)$.

This uncertainty on the initial condition is modelled
by the random variables $X_1,\ldots X_7$ that are independent and distributed as:

- $X_i\sim\mathcal{U}(-\pi,\pi)$, for $i\in\{1,2,3\}$,
- $\nu(\mathbf{X})=X_4\sim\mathcal{U}(\nu_{\min},\nu_{\max})$,
- $X_i\sim\mathcal{U}(-1,1)$, for $i\in\{5,6,7\}$.

Then,
Geraci et al. (2015) consider the integral of the temperature along the rod

$$\mathcal{M}(\mathbf{X}) = \int_{\mathcal{D}}u(x,T;\mathbf{X})dx$$

and are interested in the estimation
of its
[HeatEquationConfiguration][gemseo_umdo.use_cases.heat_equation.configuration.HeatEquationConfiguration.expectation]:

$$\mathbb{E}[\mathcal{M}(\mathbf{X})] = 50H_1+\frac{49}{4}(H_3+50H_9+50H_{21})$$

where
$H_k=\frac{2}{k^3\pi^3T}
\frac{\exp(-\nu_{\min}k^2\pi^2T)-\exp(-\nu_{\max}k^2\pi^2T)}{\nu_{\max}-\nu_{\min}}$.

The [HeatEquationModel][gemseo_umdo.use_cases.heat_equation.model.HeatEquationModel]
computes the temperature at final time
from instances of the random variables ``"X_1"``, ..., ``"X_7"``
defined over the
[HeatEquationUncertainSpace]
[gemseo_umdo.use_cases.heat_equation.uncertain_space.HeatEquationUncertainSpace].
The temperature ``"u_mesh"`` is computed at each mesh node
while the temperature ``"u"`` is an integral over the rod.

See Also:
    Geraci et al., A multifidelity control variate approach
    for the multilevel Monte Carlo technique, Center for Turbulence Research,
    Annual Research Briefs, 2015.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import abs
from numpy import array
from numpy import linspace
from numpy import meshgrid
from numpy import newaxis
from numpy import pi
from numpy import sin
from numpy import trapz

from gemseo_umdo.use_cases.heat_equation.configuration import HeatEquationConfiguration

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HeatEquationModel:
    """The discipline computing the temperature averaged over the rod at final time.

    This discipline can also compute a first-order polynomial centered at the mean input
    value.
    """

    configuration: HeatEquationConfiguration
    """The configuration of the heat equation problem."""

    taylor_mean: float
    """The expectation of the output of the first-order Taylor polynomial."""

    def __init__(
        self,
        mesh_size: int = 100,
        n_modes: int = 21,
        final_time: float = 0.5,
        nu_bounds: tuple[float, float] = (0.001, 0.009),
        rod_length: float = 1.0,
    ) -> None:
        """
        Args:
            mesh_size: The number of equispaced spatial nodes.
            n_modes: The number of modes of the truncated Fourier expansion.
            final_time: The time of interest.
            nu_bounds: The bounds of the thermal diffusivity.
            rod_length: The length of the rod.
        """  # noqa: D205 D212 D415
        self.configuration = HeatEquationConfiguration(
            mesh_size, n_modes, final_time, nu_bounds, rod_length
        )
        self.__nu_delta = nu_bounds[1] - nu_bounds[0]
        self.__modes = linspace(1, n_modes, n_modes)
        xx, nn = meshgrid(self.configuration.mesh, self.__modes, copy=False)
        self.__sinus = np.sin(xx * nn * pi)[:, :, newaxis]
        self.__default_input_value = array([0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0])
        pi_mesh = pi * self.configuration.mesh
        self.__F1 = sin(pi_mesh)  # noqa: N806
        self.__F2 = (  # noqa: N806
            sin(2 * pi_mesh)
            + sin(3 * pi_mesh)
            + 50 * (sin(9 * pi_mesh) + sin(21 * pi_mesh))
        )
        self.__term1 = self.__term2 = self.__term3 = self.__f_at_mu_X = 0
        self.__compute_taylor_materials()
        self.taylor_mean = self.__f_at_mu_X + 600 * self.__term1

    def __compute_initial_temperature(
        self,
        X: NDArray[float],  # noqa: N803
    ) -> NDArray[float]:
        """Compute the initial temperature for each mesh nodes.

        From Geraci et al., 2015 (Equation 5.2).

        Args:
            X: The input samples
                shaped as ``(sample_size, input_dimension)``.

        Returns:
            The initial temperature for each mesh nodes.
        """
        G = 50 * (4 * abs(X[:, 4:7]) - 1).T.prod(0)  # noqa: N806
        I = 3.5 * (  # noqa: N806, E741
            sin(X[:, 0]) + 7 * sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * sin(X[:, 0])
        )
        return (
            self.__F1[:, newaxis] * G[newaxis, :]
            + self.__F2[:, newaxis] * I[newaxis, :]
        )

    def __call__(
        self, input_samples: NDArray[float] | None = None, batch_size: int = 50000
    ) -> tuple[NDArray[float] | float, NDArray[float]]:
        """Compute the temperature.

        Args:
            input_samples: The input samples
                shaped as ``(sample_size, input_dimension)`` or ``(input_dimension, )``.
            batch_size: The maximum number of samples per batch.

        Returns:
            - The integrated temperature shaped as ``(sample_size, )`` or ``()``.
            - The temperature at the different nodes
                shaped as ``(sample_size, n_nodes)`` or ``(n_nodes, )``.
        """
        if input_samples is None:
            input_samples = self.__default_input_value

        if input_samples.ndim == 1:
            is_input_samples_1d = True
            input_samples = input_samples[newaxis, :]
        else:
            is_input_samples_1d = False

        n_samples = len(input_samples)
        if n_samples <= batch_size:
            u, u_mesh = self.__evaluate(input_samples)
        else:
            u = np.zeros(n_samples)
            u_mesh = np.zeros((n_samples, self.configuration.mesh_size))
            i_start = 0
            while n_samples > 0:
                n_samples_batch = min(batch_size, n_samples)
                indices = slice(i_start, i_start + n_samples_batch)
                u[indices], u_mesh[indices] = self.__evaluate(input_samples[indices])
                i_start += n_samples_batch
                n_samples -= n_samples_batch

        if is_input_samples_1d:
            return u[0], u_mesh[0]

        return u, u_mesh

    def __evaluate(
        self,
        X: NDArray[float],  # noqa: N803
    ) -> tuple[NDArray[float], NDArray[float]]:
        """Compute the temperature.

        From Geraci et al., 2015 (Equation 5.4).

        Args:
            X: The input samples
                shaped as ``(sample_size, input_dimension)``.

        Returns:
            The integrated temperature shaped as ``(sample_size, )``,
            the temperature at the different nodes shaped as ``(sample_size, n_nodes)``.
        """
        term = np.trapz(
            self.__sinus * self.__compute_initial_temperature(X)[newaxis, :, :],
            x=self.configuration.mesh,
            axis=1,
        ) * np.exp(
            -X[:, 3][newaxis, :]
            * (self.__modes[:, newaxis] * pi) ** 2
            * self.configuration.final_time
        )
        u_mesh = 2 * np.sum(self.__sinus * term[:, newaxis, :], axis=0)
        return trapz(u_mesh, x=self.configuration.mesh, axis=0), u_mesh.T

    def __compute_taylor_materials(self) -> None:
        """Compute the materials of the first-order Taylor polynomial."""
        mu_X = self.__default_input_value  # noqa: N806
        x = self.configuration.mesh
        n = self.__modes
        sn = self.__sinus

        u0_at_mu_X = self.__compute_initial_temperature(  # noqa: N806
            mu_X[newaxis, :]
        ).reshape(-1)  # -> (nx, 1) => (nx,)
        snu0_at_mu_X = sn[:, :, 0] * u0_at_mu_X[None, :]  # -> (n_modes, nx)# noqa: N806
        snF1 = sn[:, :, 0] * self.__F1[None, :]  # noqa: N806 -> (n_modes, nx)
        snF2 = sn[:, :, 0] * self.__F2[None, :]  # noqa: N806 -> (n_modes, nx)

        sn_quad = np.trapz(sn, x=x, axis=1).ravel()  # -> (n_modes,)
        snF1_quad = np.trapz(snF1, x=x, axis=1)  # noqa: N806 -> (n_modes,)
        snF2_quad = np.trapz(snF2, x=x, axis=1)  # noqa: N806 -> (n_modes,)
        A_n_at_mu_X_quad = 2 * np.trapz(snu0_at_mu_X, x=x, axis=1)  # noqa: N806
        # -> (n_modes,)
        B_n_at_mu_X_quad = (  # noqa: N806
            np.exp(-mu_X[3] * (n * np.pi) ** 2 * 0.5) * sn_quad
        )  # -> (n_modes,)

        self.__term1 = np.sum(B_n_at_mu_X_quad * snF1_quad)  # (scalar)
        self.__term2 = np.sum(B_n_at_mu_X_quad * snF2_quad)  # (scalar)
        self.__term3 = np.sum(  # (scalar)
            A_n_at_mu_X_quad * sn_quad * n**2 * np.exp(-mu_X[3] * n**2 * np.pi**2 * 0.5)
        )
        self.__f_at_mu_X = self(mu_X)[0]  # -> (1,) => (scalar)

    def compute_taylor(self, input_samples: NDArray[float]) -> NDArray[float]:
        """Evaluate the first-order Taylor polynomial.

        Args:
            input_samples: The input samples
                shaped as ``(sample_size, input_dimension)`` or ``(input_dimension, )``.

        Returns:
            The output samples of the first-order Taylor polynomial
            shaped as ``(sample_size, n_nodes)`` or ``(n_nodes, )``.
        """
        X = input_samples  # noqa: N806
        mu_X = self.__default_input_value  # noqa: N806
        return self.__f_at_mu_X + (
            7 * X[..., [0]] * self.__term2
            - (X[..., [3]] - mu_X[3]) * np.pi**2 * 0.5 * self.__term3
            + 400
            * self.__term1
            * (np.abs(X[..., [4]]) + np.abs(X[..., [5]]) + np.abs(X[..., [6]]))
        )
