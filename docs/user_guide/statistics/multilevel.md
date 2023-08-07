<!---
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
--->

# Multilevel Monte Carlo

A classical problem consists of
estimating a statistic $\theta$ of the output of a simulator $f$
whose input $\mathbf{X}$ is random:
that is, a statistic $\theta$ of $Y=f(\mathbf{X})$.

## Multilevel Monte Carlo methods

Crude Monte Carlo (MC) is the most standard method
to estimate it.
For instance,
given a sample $\left(\mathbf{X}^{(1)},\ldots,\mathbf{X}^{(N)}\right)$
made of independent random variables distributed as $\mathbf{X}$,
$\frac{1}{N}\sum_{i=1}^Nf(\mathbf{X}^{(i)})$ is
an unbiased MC estimator of the expectation of $Y$
whose variance is $\mathcal{O}(N^{-1})$.
(N.B. by unbiasedness, the variance of the MC estimator equals its mean squared error.)

In presence of a sequence of simulators $(f_\ell)_{\ell = 0}^L$
with increasing accuracy and computational cost,
such that $f_L = f$,
multilevel Monte Carlo (MLMC) methods[@giles2015multilevel] can be relevant
to reduce the variance of the MC estimator.
The MLMC methods use all these models
to estimate the statistic $\theta_L$ (a.k.a. $\theta$)
of the random output variable $f_L(\mathbf{X})$.

We denote by $Y_\ell=f_\ell(\mathbf{X})$ the random output variable
associated with the model level $f_\ell$
and by $(\theta_1,\ldots,\theta_L)$ the sequence of statistics
increasingly close to $\theta_L$,
where $\theta_\ell$ is the statistic of $Y_\ell$.
Then,
the statistical measure $\theta_L$ can be expressed as a telescoping sum
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

$$
\hat{\theta}_L^{\mathrm{MLMC}}
= \sum \limits_{\ell = 0}^{L} \hat{T}_{\ell,n_\ell}^{\mathrm{MC}}
= \sum \limits_{\ell = 0}^{L} \hat{\theta}_{\ell,n_\ell}^{\mathrm{MC},(\ell)}-
\hat{\theta}_{\ell-1,n_\ell}^{\mathrm{MC},(\ell)}.
$$

Many algorithms distributing the sampling budget between the different levels
can be found in the literature.
Their goal is often to get an MLMC estimator with an accuracy target set by the user.
Recently,
an allocation algorithm[@mycek2019multilevel] has been proposed
to get the best accuracy for a given sampling budget.

## Multilevel Monte Carlo with control variates

When surrogate models are available
as functions of the random input $\mathbf{X}$,
their random outputs can be used as control variates (CVs)
to reduce the variance of the MC estimators.

In 2023,
El Amri _et al_[@amri2023multilevel]
have investigated the combination of CVs and MLMC in different ways
and called the resulting algorithm MLMC-MLCV,
standing for _multilevel Monte Carlo with multilevel control variates_.
Their idea was to estimate each $T_\ell$ of the telescoping sum
with a control variate approach based on surrogate models.
They showed that even with surrogate models
that are moderately correlated to the original models,
the reduction in variance could be significant.
