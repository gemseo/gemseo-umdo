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
from __future__ import annotations

from numpy import array
from numpy.testing import assert_allclose

from gemseo_umdo.use_cases.truss.discipline import TrussDiscipline
from gemseo_umdo.use_cases.truss.model import TrussModel


def test_default():
    """Check TrussDiscipline with default values."""
    discipline = TrussDiscipline()
    assert discipline.io.input_grammar.descriptions == {
        "E1": "The Young's modulus of the horizontal bar #1 (in Pa).",
        "E2": "The Young's modulus of the vertical bar #1 (in Pa).",
        "A1": "The cross-sectional area of the horizontal bar #1 (in m²).",
        "A2": "The cross-sectional area of the vertical bar #1 (in m²).",
        "P1": "The vertical force #1 (in N).",
        "P2": "The vertical force #2 (in N).",
        "P3": "The vertical force #3 (in N).",
        "P4": "The vertical force #4 (in N).",
        "P5": "The vertical force #5 (in N).",
        "P6": "The vertical force #6 (in N).",
    }
    assert discipline.io.output_grammar.descriptions == {
        "V1": "The vertical displacement (in m) at the bottom central node.",
        "displacements": (
            "The nodal displacements (in m) shaped as `(n_nodes, 2)` "
            "(the first and second columns represent "
            "the horizontal and vertical displacements respectively)."
        ),
    }
    assert discipline.default_grammar_type == discipline.GrammarType.SIMPLE
    assert tuple(discipline.io.input_grammar.names) == (
        "E1",
        "E2",
        "A1",
        "A2",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
    )
    assert tuple(discipline.io.output_grammar.names) == ("V1", "displacements")
    truss = TrussModel()
    v1, displacements = truss.compute()
    discipline.execute()
    assert_allclose(discipline.io.data["V1"], array([v1]))
    assert_allclose(discipline.io.data["displacements"], displacements.ravel())


def test_custom():
    """Check TrussDiscipline with custom values."""
    discipline = TrussDiscipline(use_different_bars=True)
    assert tuple(discipline.io.input_grammar.names) == (
        "E1_1",
        "E1_2",
        "E1_3",
        "E1_4",
        "E1_5",
        "E1_6",
        "E1_7",
        "E1_8",
        "E1_9",
        "E1_10",
        "E1_11",
        "E2_1",
        "E2_2",
        "E2_3",
        "E2_4",
        "E2_5",
        "E2_6",
        "E2_7",
        "E2_8",
        "E2_9",
        "E2_10",
        "E2_11",
        "E2_12",
        "A1_1",
        "A1_2",
        "A1_3",
        "A1_4",
        "A1_5",
        "A1_6",
        "A1_7",
        "A1_8",
        "A1_9",
        "A1_10",
        "A1_11",
        "A2_1",
        "A2_2",
        "A2_3",
        "A2_4",
        "A2_5",
        "A2_6",
        "A2_7",
        "A2_8",
        "A2_9",
        "A2_10",
        "A2_11",
        "A2_12",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
    )
    assert discipline.io.input_grammar.descriptions == {
        "E1_1": "The Young's modulus of the horizontal bar #1 (in Pa).",
        "E1_2": "The Young's modulus of the horizontal bar #2 (in Pa).",
        "E1_3": "The Young's modulus of the horizontal bar #3 (in Pa).",
        "E1_4": "The Young's modulus of the horizontal bar #4 (in Pa).",
        "E1_5": "The Young's modulus of the horizontal bar #5 (in Pa).",
        "E1_6": "The Young's modulus of the horizontal bar #6 (in Pa).",
        "E1_7": "The Young's modulus of the horizontal bar #7 (in Pa).",
        "E1_8": "The Young's modulus of the horizontal bar #8 (in Pa).",
        "E1_9": "The Young's modulus of the horizontal bar #9 (in Pa).",
        "E1_10": "The Young's modulus of the horizontal bar #10 (in Pa).",
        "E1_11": "The Young's modulus of the horizontal bar #11 (in Pa).",
        "E2_1": "The Young's modulus of the vertical bar #1 (in Pa).",
        "E2_2": "The Young's modulus of the vertical bar #2 (in Pa).",
        "E2_3": "The Young's modulus of the vertical bar #3 (in Pa).",
        "E2_4": "The Young's modulus of the vertical bar #4 (in Pa).",
        "E2_5": "The Young's modulus of the vertical bar #5 (in Pa).",
        "E2_6": "The Young's modulus of the vertical bar #6 (in Pa).",
        "E2_7": "The Young's modulus of the vertical bar #7 (in Pa).",
        "E2_8": "The Young's modulus of the vertical bar #8 (in Pa).",
        "E2_9": "The Young's modulus of the vertical bar #9 (in Pa).",
        "E2_10": "The Young's modulus of the vertical bar #10 (in Pa).",
        "E2_11": "The Young's modulus of the vertical bar #11 (in Pa).",
        "E2_12": "The Young's modulus of the vertical bar #12 (in Pa).",
        "A1_1": "The cross-sectional area of the horizontal bar #1 (in m²).",
        "A1_2": "The cross-sectional area of the horizontal bar #2 (in m²).",
        "A1_3": "The cross-sectional area of the horizontal bar #3 (in m²).",
        "A1_4": "The cross-sectional area of the horizontal bar #4 (in m²).",
        "A1_5": "The cross-sectional area of the horizontal bar #5 (in m²).",
        "A1_6": "The cross-sectional area of the horizontal bar #6 (in m²).",
        "A1_7": "The cross-sectional area of the horizontal bar #7 (in m²).",
        "A1_8": "The cross-sectional area of the horizontal bar #8 (in m²).",
        "A1_9": "The cross-sectional area of the horizontal bar #9 (in m²).",
        "A1_10": "The cross-sectional area of the horizontal bar #10 (in m²).",
        "A1_11": "The cross-sectional area of the horizontal bar #11 (in m²).",
        "A2_1": "The cross-sectional area of the vertical bar #1 (in m²).",
        "A2_2": "The cross-sectional area of the vertical bar #2 (in m²).",
        "A2_3": "The cross-sectional area of the vertical bar #3 (in m²).",
        "A2_4": "The cross-sectional area of the vertical bar #4 (in m²).",
        "A2_5": "The cross-sectional area of the vertical bar #5 (in m²).",
        "A2_6": "The cross-sectional area of the vertical bar #6 (in m²).",
        "A2_7": "The cross-sectional area of the vertical bar #7 (in m²).",
        "A2_8": "The cross-sectional area of the vertical bar #8 (in m²).",
        "A2_9": "The cross-sectional area of the vertical bar #9 (in m²).",
        "A2_10": "The cross-sectional area of the vertical bar #10 (in m²).",
        "A2_11": "The cross-sectional area of the vertical bar #11 (in m²).",
        "A2_12": "The cross-sectional area of the vertical bar #12 (in m²).",
        "P1": "The vertical force #1 (in N).",
        "P2": "The vertical force #2 (in N).",
        "P3": "The vertical force #3 (in N).",
        "P4": "The vertical force #4 (in N).",
        "P5": "The vertical force #5 (in N).",
        "P6": "The vertical force #6 (in N).",
    }
    assert tuple(discipline.io.output_grammar.names) == ("V1", "displacements")
    truss = TrussModel()
    v1, displacements = truss.compute()
    discipline.execute()
    assert_allclose(discipline.io.data["V1"], array([v1]))
    assert_allclose(discipline.io.data["displacements"], displacements.ravel())
