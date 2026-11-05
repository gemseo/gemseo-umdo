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
r"""Truss problem.

This problem from the literature is a truss structure
comprising 11 horizontal bars,
with uncertain Young's modulus $E_1$ and cross-section area $A_1$,
and 12 oblical bars,
with uncertain Young's modulus $E_2$ and cross-section area $A_2$.
The 6 upper nodes are subjected to uncertain vertical loads $P_1,\ldots,P_6$
and the variable of interest is the vertical displacement $V_1$
at the bottom central node.

!!! quote "References"
    Géraud Blatman, Bruno Sudret,
    *Adaptive sparse polynomial chaos expansion based on least angle regression*,
    Journal of Computational Physics,
    Volume 230, Issue 6,
    2011,
    Pages 2345-2367,
    [https://doi.org/10.1016/j.jcp.2010.12.021](https://doi.org/10.1016/j.jcp.2010.12.021).

    Sang Hoon Lee, Byung Man Kwak,
    *Response surface augmented moment method for efficient reliability analysis*,
    Structural Safety,
    Volume 28, Issue 3,
    2006,
    Pages 261-272,
    [https://doi.org/10.1016/j.strusafe.2005.08.003](https://doi.org/10.1016/j.strusafe.2005.08.003).
"""
