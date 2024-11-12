<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# gemseo-umdo

[![PyPI - License](https://img.shields.io/pypi/l/gemseo-umdo)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gemseo-umdo)](https://pypi.org/project/gemseo-umdo/)
[![PyPI](https://img.shields.io/pypi/v/gemseo-umdo)](https://pypi.org/project/gemseo-umdo/)
[![Codecov branch](https://img.shields.io/codecov/c/gitlab/gemseo:dev/gemseo-umdo/develop)](https://app.codecov.io/gl/gemseo:dev/gemseo-umdo)

## Overview

`gemseo-umdo` is a plugin of the library [GEMSEO](https://www.gemseo.org),
dedicated to multidisciplinary optimization (MDO) under uncertainty.

### MDO under uncertainty

The main goal of `gemseo-umdo` is to extend GEMSEO
to MDO under uncertainty.

Given a collection of disciplines,
we are interested in solving a problem like

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}&
& \mathbb{E}[f(x,U)]+\kappa\times\mathbb{S}[f(x,U)] \\
&\operatorname{subject\;to}
& &\mathbb{P}[g(x,U)\geq 0] \leq \varepsilon
\end{align}
$$

by selecting an MDO formulation to handle the multidisciplinary coupling
and an estimation technique to approximate the statistics.

### Statistics

`gemseo-umdo` also proposes advanced techniques
for uncertainty quantification and management (UQ&M).
In presence of multilevel simulators,
multilevel Monte Carlo (MLMC) sampling can reduce
the variance of the statistics estimators.
Another variance reduction technique
consists of using the outputs of surrogate models
as control variates,
even moderately correlated with the original models.

### Visualization

A third facet of `gemseo-umdo` is the visualization toolbox
to display the propagation of the uncertainties
through a multidisciplinary system
as well as the interaction between the uncertain input variables.

## Installation

Install the latest version with `pip install gemseo-umdo`.

See [pip](https://pip.pypa.io/en/stable/getting-started/) for more information.

## Bugs and questions

Please use the [gitlab issue tracker](https://gitlab.com/gemseo/dev/gemseo-umdo/-/issues)
to submit bugs or questions.

## Contributing

See the [contributing section of GEMSEO](https://gemseo.readthedocs.io/en/stable/software/developing.html#dev).

## Contributors

- Antoine Dechaume
- Matthias De Lozzo
