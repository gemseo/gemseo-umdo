<!---
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
--->

# Getting started

GEMSEO-UMDO is an extension of the open-source library [GEMSEO](https://www.gemseo.org)
which is dedicated to multidisciplinary optimization (MDO).
This extension is also open-source,
under the [LGPL v3 license](https://www.gnu.org/licenses/lgpl-3.0.en.html).

[Installation](user_guide/installation.md){ .md-button }

## MDO under uncertainty

The main goal of GEMSEO-UMDO is to extend GEMSEO
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

[Read more](user_guide/umdo/index.md){ .md-button }
[Examples](generated/examples/umdo/index.md){ .md-button }

## Statistics

GEMSEO-UMDO also proposes advanced techniques
for uncertainty quantification and management (UQ&M).
In presence of multilevel simulators,
multilevel Monte Carlo (MLMC) sampling can reduce
the variance of the statistics estimators.
Another variance reduction technique
consists of using the outputs of surrogate models
as control variates,
even moderately correlated with the original models.

[Read more](user_guide/statistics/index.md){ .md-button }

## Visualization

A third facet of GEMSEO-UMDO is the visualization toolbox
to display the propagation of the uncertainties
through a multidisciplinary system
as well as the interaction between the uncertain input variables.

[Read more](user_guide/visualization/index.md){ .md-button }
[Examples](generated/examples/visualizations/index.md){ .md-button }
