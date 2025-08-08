<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Control variate

[ControlVariate][gemseo_umdo.formulations.control_variate.ControlVariate]
is a U-MDO formulation that estimates the statistics
using control variates based on either first-order Taylor polynomials (default) or regression models,
a.k.a. surrogate models.

!!! note "Control variates (CVs) method"

    The control variates method is a variance reduction technique used in Monte Carlo sampling.
    [Read more](https://en.wikipedia.org/wiki/Control_variates)

This U-MDO formulation has one mandatory parameter, namely `n_samples`,
which corresponds to the number of Monte Carlo samples in the CVs method.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation_settings=ControlVariate_Settings(n_samples=20),
)
```

## Settings

### Control variates

The control variates are built from approximations of the original objective, constraint and observable functions.
These approximations can be
either Taylor polynomials centered at the mean input value (default)
or regression models defined as [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor]s.
In the latter case,
the regression model is defined by the Pydantic models
`regressor_settings` for the settings of the regressor
(e.g. [RBFRegressor_Settings][gemseo.mlearning.regression.algos.rbf_settings.RBFRegressor_Settings])
and `regressor_doe_algo_settings` for the settings of the DOE algorithm used to create the training dataset
(e.g. [OT_HALTON_Settings][gemseo.algos.doe.openturns.settings.ot_halton.OT_HALTON_Settings]).

### Sampling

By default,
the formulation uses the DOE algorithm `OT_OPT_LHS` with 10 samples:
the Latin hypercube sampling (LHS)
enhanced by simulated annealing
of OpenTURNS.
Simulated annealing is a global optimization technique that
starts from an initial LHS
and improves it to maximize its discrepancy
and so to get a better space-filling LHS.

The number of samples can be changed with the parameter `n_samples`
and the DOE algorithm name can be changed with the parameter `doe_algo_settings` ,
which is a Pydantic model deriving from [BaseDOESettings][gemseo.algos.doe.base_doe_settings.BaseDOESettings].
When `n_samples` is `None` (default) and `doe_algo_settings` has a field `n_samples`,
then this field is considered.
When `doe_algo_settings` has a field `seed` and its value is `None`,
then the U-MDO formulation will use [SEED][gemseo.utils.seeder.SEED].

!!! note "API"
    Here is an example of code that considers an RBF regressor and different DOE algorithms:

    ``` py
    settings = ControlVariate_Settings(
        doe_algo_settings=OT_MONTE_CARLO(n_samples=20, n_processes=2),
        regressor_settings=RBFRegressor_Settings(),
        regressor_doe_algo_settings=OT_OPT_LHS(n_samples=15),
    )
    scenario = UMDOScenario(
        disciplines,
        mdo_formulation_name,
        objective_name,
        design_space,
        uncertain_space,
        statistic_name,
        statistic_estimation_settings=settings,
    )
    ```

    !!! warning
        `doe_algo_settings` and `regressor_doe_algo_settings` must use different `seed`
        when their classes are the same, e.g.
        ``` py
        settings = ControlVariate_Settings(
            doe_algo_settings=OT_MONTE_CARLO(n_samples=20, n_processes=2, seed=2),
            regressor_settings=RBFRegressor_Settings(),
            regressor_doe_algo_settings=OT_MONTE_CARLO(n_samples=15, seed=3),
        )
        ```

## Statistics

This U-MDO formulation has been implemented
for the expectation, the standard deviation, the variance, the margin and the probability.

Only the formula for the expectation is noted here, for simplicity's sake.

### Taylor polynomial

In the case of Taylor polynomial,
the control variate estimator of the expectation is:

$$\mathbb{E}[f(x,U)]
\approx
\frac{1}{N}\sum_{i=1}^N f\left(x,u^{(i)}\right)
+\alpha_N\left(\frac{1}{N}\sum_{j=1}^N \tilde{f}\left(x,u^{(j)}\right)-f(x,\mu)\right)$$

where $\tilde{f}(x,u)$ is the first-order Taylor polynomial of $f(x,u)$ at $u=\mu$,
$\alpha_N$ is the empirical estimator
of $\frac{\text{cov}\left[f(x,U),\tilde{f}(x,U)\right]}
{\mathbb{V}\left[f(x,U)\right]}$
and $u^{(1)},\ldots,u^{(N)}$ are $N$ independent realizations of $U$.

### Regression models

In the case of a regression model,
the control variate estimator of the expectation is:

$$\mathbb{E}[f(x,U)]
\approx
\frac{1}{N}\sum_{i=1}^N f\left(x,u^{(i)}\right)
+\alpha_N\left(\frac{1}{N}\sum_{j=1}^N \hat{f}\left(x,u^{(j)}\right)-\frac{1}{M}\sum_{j=1}^M \hat{f}\left(x,u^{(N+j)}\right)\right)$$

where $\tilde{f}(x)$ is the regression model of $f(x)$,
$\alpha_N$ is the empirical estimator
of $\frac{\text{cov}\left[f(x,U),\hat{f}(x,U)\right]}
{\mathbb{V}\left[f(x,U)\right]}$,
$u^{(1)},\ldots,u^{(N)},u^{(N+1)},\ldots,u^{(N+M)}$ are $N+M$ independent realizations of $U$
with $M\gg 1$.
