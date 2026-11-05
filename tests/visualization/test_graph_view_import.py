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

import sys
import warnings
from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "gemseo_umdo.visualization.sobol_graph",
        "gemseo_umdo.visualization.uncertain_coupling_graph",
    ],
)
def test_no_deprecated_graph_view_import(monkeypatch, module_name):
    """Check that importing a module does not use `gemseo.post._graph_view`."""
    package_name, attribute_name = module_name.rsplit(".", 1)
    monkeypatch.delitem(sys.modules, "gemseo.post._graph_view", raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delattr(import_module(package_name), attribute_name, raising=False)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        import_module(module_name)

    assert not [r for r in records if "_graph_view" in str(r.message)]
