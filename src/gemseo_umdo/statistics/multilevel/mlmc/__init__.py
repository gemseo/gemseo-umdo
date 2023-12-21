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
r"""Multilevel Monte Carlo (MLMC) algorithm.

The goal of the MLMC algorithm is to estimate a statistic $\theta$
(ex: mean, variance)
of the output of a simulator $f$ whose input $\mathbf{X}$ is random:
that is, a statistic $\theta$ of $Y=f(\mathbf{X})$.

Let $(f_\ell)_{\ell = 0}^L$ be a sequence of model levels
with increasing accuracy and computational cost,
such that $f_L = f$.
The MLMC algorithm uses all these models
to estimate the statistic $\theta_L$ (a.k.a. $\theta$)
of the random output variable $f_L(\mathbf{X})$
where $\mathbf{X}$ is a random input vector.

We denote by $Y_\ell=f_\ell(\mathbf{X})$ the random output variable
associated with the model level $f_\ell$
and by $(\theta_\ell)_{\ell = 0}^L$ the sequence of statistics
increasingly *close* to $\theta_L$
where $\theta_\ell$ is the statistic of $Y_\ell$.

The statistical measure $\theta_L$ can be expressed as a telescoping sum
$\theta_L = \sum \limits_{\ell = 0}^{L} T_\ell$,
where $T_\ell = \theta_\ell - \theta_{\ell-1}$,
and by convention $\theta_{-1} = 0$.

Let $\hat{\theta}_{\ell,n_\ell}^{\mathrm{MC},(\ell)}$
and $\hat{\theta}_{\ell-1,n_\ell}^{\mathrm{MC},(\ell)}$ be respectively
the Monte Carlo (MC) estimators of $\theta_\ell$ and $\theta_{\ell-1}$
using the same $n_{\ell}$-sample.

Then,
the MLMC estimator $\hat{\theta}_L^{\mathrm{ML}}$ of $\theta_L$
may be expressed as:

$$\hat{\theta}_L^{\mathrm{MLMC}}
= \sum \limits_{\ell = 0}^{L} \hat{T}_{\ell,n_\ell}^{\mathrm{MC}}
= \sum \limits_{\ell = 0}^{L} \hat{\theta}_{\ell,n_\ell}^{\mathrm{MC},(\ell)}
- \hat{\theta}_{\ell-1,n_\ell}^{\mathrm{MC},(\ell)}.
$$
"""

from __future__ import annotations
