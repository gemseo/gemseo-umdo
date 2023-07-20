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
r"""Multi-level Monte Carlo (MLMC) algorithm.

The goal of the MLMC algorithm is to estimate a statistic :math:`theta`
(ex: mean, variance)
of the output of a simulator :math:`f` whose input :math:`\mathbf{X}` is random:
that is, a statistic :math:`theta` of :math:`Y=f(\mathbf{X})`.

Let :math:`(f_\ell)_{\ell = 0}^L` be a sequence of model levels
with increasing accuracy and computational cost,
such that :math:`f_\ell = f`.
The MLMC algorithm uses all these models
to estimate the statistic :math:`\theta_L` (a.k.a. :math:`theta`)
of the random output variable :math:`f_L(\mathbf{X})`
where :math:`\mathbf{X}` is a random input vector.

We denote by :math:`Y_\ell=f_\ell(\mathbf{X})` the random output variable
associated with the model level :math:`f_\ell`
and by :math:`(\theta_\ell)_{\ell = 0}^L` the sequence of statistics
increasingly *close* to :math:`\theta_L`
where :math:`\theta_\ell` is the statistic of :math:`Y_\ell`.

The statistical measure :math:`\theta_L` can be expressed as a telescoping sum:

.. math::

   \theta_L = \sum \limits_{\ell = 0}^{L} T_\ell,

where :math:`T_\ell = \theta_\ell - \theta_{\ell-1}`,
and by convention :math:`\theta_{-1} = 0`.

Let :math:`\hat{\theta}_{\ell,n_\ell}^{\mathrm{MC},(\ell)}`
and :math:`\hat{\theta}_{\ell-1,n_\ell}^{\mathrm{MC},(\ell)}` be respectively
the Monte Carlo (MC) estimators of :math:`\theta_\ell` and :math:`\theta_{\ell-1}`
using the same :math:`n_{\ell}`-sample.

Then,
the MLMC estimator :math:`\hat{\theta}_L^{\mathrm{ML}}` of :math:`\theta_L`
may be expressed as:

.. math::

   \hat{\theta}_L^{\mathrm{ML}}
   = \sum \limits_{\ell = 0}^{L} \hat{T}_{\ell,n_\ell}^{\mathrm{MC}}
   = \sum \limits_{\ell = 0}^{L} \hat{\theta}_{\ell,n_\ell}^{\mathrm{MC},(\ell)}
   - \hat{\theta}_{\ell-1,n_\ell}^{\mathrm{MC},(\ell)}.
"""
from __future__ import annotations
