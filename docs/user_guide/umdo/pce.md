<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Polynomial chaos expansion

[PCE][gemseo_umdo.formulations.pce.PCE]
is a U-MDO formulation that estimates the statistics
using polynomial chaos expansions (PCEs).

At each iteration of the optimization loop,
a PCE is built over the uncertain space
and its coefficients are used to estimate specific statistics,
namely mean, standard deviation, variance and margin.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation_settings=PCE_Settings(doe_n_samples=20),
)
```

## Settings

### DOE algorithm

By default,
the formulation uses the [OpenTURNS](https://openturns.github.io)' DOE algorithm `OT_OPT_LHS`,
which is a Latin hypercube sampling (LHS) technique enhanced by simulated annealing,
a global optimization technique that
starts from an initial LHS
and improves it to maximize its discrepancy
and so to get a better space-filling LHS.

The default number samples is 10.
It can be changed with the parameter `n_samples`
and the DOE algorithm name can be changed with the parameter `doe_algo_settings`,
which is a Pydantic model deriving from [BaseDOESettings][gemseo.algos.doe.base_doe_settings.BaseDOESettings].
When `n_samples` is `None` (default) and `doe_algo_settings` has a field `n_samples`,
then this field is considered.
When `doe_algo_settings` has a field `seed` and its value is `None`,
then the U-MDO formulation will use [SEED][gemseo.utils.seeder.SEED].

### PCE's options

This U-MDO formulation is based on the [PCERegressor][gemseo.mlearning.regression.algos.pce.PCERegressor] available in GEMSEO,
which wraps the [OpenTURNS' PCE algorithm](https://openturns.github.io/openturns/latest/user_manual/response_surface/_generated/openturns.FunctionalChaosAlgorithm.html).
Use the `regressor_settings` parameter to set the options of the [PCERegressor][gemseo.mlearning.regression.algos.pce.PCERegressor],
using the Pydantic model [PCERegressorSettings][gemseo.mlearning.regression.algos.pce_settings.PCERegressor_Settings].
For example,
set `use_lars` to `True` to obtain a more sparse PCE and avoid overfitting
([more details](https://openturns.github.io/openturns/latest/theory/meta_modeling/polynomial_sparse_least_squares.html))
and `degree` to `3` for a maximum degree of 3.
You can also use the technique proposed in Section II.C.3 of a paper by Mura _et al._[@Mura2020]
to approximate the Jacobians of the mean, standard deviation and variance with respect to the design variables
at no extra cost,
if you do not want to compute the derivatives of the disciplines to reduce the calculation budget
or approximate these Jacobians by finite differences.
You only have to enable the option `approximate_statistics_jacobians`.

!!! note "API"
    Use `statistic_estimation_settings`
    to set the DOE algorithm and the PCE's settings,
    e.g.

    ``` py
    settings = PCE_Settings(
        doe_algo_settings=OT_MONTE_CARLO(n_samples=20, n_processes=2),
        regressor_settings=PCE_Settings.PCERegressorSettings(use_lars=True, degree=3),
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

### Quality options

Finally,
many options can be used
to adjust the log verbosity
providing information on the PCEs built at each optimization iteration:

| Name                 | Description                                                                                   |
|----------------------|-----------------------------------------------------------------------------------------------|
| quality_threshold    | The learning quality threshold below which a warning is logged.                               |
| quality_name         | The name of the measure to assess the quality of the PCE regressor.                           |
| quality_cv_compute   | Whether to estimate the quality by cross-validation (CV).                                     |
| quality_n_folds      | The number of folds in the case of the CV technique.                                          |
| quality_cv_randomize | Whether to shuffle the samples before dividing them in folds in the case of the CV technique. |
| quality_cv_seed      | The seed of the pseudo-random number generator.                                               |
| quality_cv_threshold | The CV quality threshold below which a warning is logged.                                     |

## Statistics

This U-MDO formulation has been implemented
for the expectation, the standard deviation, the variance and the margin,
from the coefficients $(\alpha_i)_{0\leq i \leq N}$ of the PCE

$$\hat{f}_x(U)=\alpha_0 + \sum_{1\leq i\leq P}\alpha_i\Phi_i(U).$$

| Statistic          | Notation                        | Estimator                                                                                                                    |
|--------------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Mean               | $\mathbb{E}[\varphi(x,U)]$      | $E_{\textrm{PCE}}[\varphi(x,U)]=\alpha_0$                                                                                    |
| Variance           | $\mathbb{V}[\varphi(x,U)]$      | $V_{\textrm{PCE}}[\varphi(x,U)]=\sum_{1\leq i\leq P}\alpha_i^2$                                                              |
| Standard deviation | $\mathbb{S}[\varphi(x,U)]$      | $S_{\textrm{PCE}}[\varphi(x,U)]=\sqrt{V_{\textrm{PCE}}[\varphi(x,U)]}$                                                      |
| Margin             | $\textrm{Margin}[\varphi(x,U)]$ | $\textrm{Margin}_{\textrm{PCE}}[\varphi(x,U)]=E_{\textrm{PCE}}[\varphi(x,U)] + \kappa \cdot S_{\textrm{PCE}}[\varphi(x,U)]$ |

## Gradient-based optimization

When the multidisciplinary process is differentiable,
and a gradient-based optimizer is used,
analytical derivatives are implemented with the following statistics:
mean, standard deviation, variance and margin.
