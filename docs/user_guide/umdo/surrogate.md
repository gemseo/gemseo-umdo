<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Surrogate

[Surrogate][gemseo_umdo.formulations.surrogate.Surrogate]
is a U-MDO formulation that estimates the statistics
using surrogate models.

At each iteration of the optimization loop,
a surrogate model is built over the uncertain space
and Monte Carlo sampling is used to estimate specific statistics,
namely mean, standard deviation, variance, probability and margin.

The quality of the surrogate model is logged
and stored in the database attached to the scenario
(see `scenario.formulation.optimization_problem.database`).

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation_settings=Surrogate_Settings(doe_n_samples=20),
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

### Surrogate's options

This U-MDO formulation is based on a [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
By default,
this surrogate model is the [RBFRegressor][gemseo.mlearning.regression.algos.rbf.RBFRegressor] available in GEMSEO,
which wraps the [SciPy's RBF algorithm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html).
The kind of regressor can be changed
by setting the `regressor_settings` parameter with a Pydantic model defining the settings of a regressor
For example,
set `regressor_settings` to `"LinearRegressor_Settings()"` to use a linear regressor,
`regressor_settings` to `RBFRegressor_Settings(function="cubic")` for a
[RBFRegressor][gemseo.mlearning.regression.algos.rbf.RBFRegressor] based on a cubic function
and `regressor_n_samples` to `100` to estimate the statistics with 100 Monte Carlo simulations instead of 10000.

!!! note "API"
    Use `statistic_estimation_settings`
    to set the DOE algorithm and the surrogate's options,
    e.g.

    ``` py
    settings = Surrogate_Settings(
        doe_algo_settings=OT_MONTE_CARLO_Settings(n_samples=20, n_processes=2),
        regressor_settings=PolynomialRegressor_Settings(degree=3),
        regressor_n_samples=100,
    )
    scenario = UMDOScenario(
        disciplines,
        mdo_formulation_name,
        objective_name,
        design_space,
        uncertain_space,
        statistic_name,
        statistic_estimation_parameters=settings,
    )
    ```

### Quality options

Finally,
many options can be used
to adjust the log verbosity
providing information on the surrogates built at each optimization iteration:

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
for the expectation, the standard deviation, the variance, the margin and the probability,
by sampling the surrogate model $\widehat{\varphi}$ of $\varphi$.

| Statistic          | Notation                         | Estimator                                                                                                                                         |
|--------------------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Mean               | $\mathbb{E}[\varphi(x,U)]$       | $E_{\textrm{Surrogate}}[\varphi(x,U)]=\frac{1}{N}\sum_{i=1}^N\widehat{\varphi}(x,u^{(i)})$                                                        |
| Variance           | $\mathbb{V}[\varphi(x,U)]$       | $V_{\textrm{Surrogate}}[\varphi(x,U)]=\frac{1}{N-1}\sum_{i=1}^N\left(\widehat{\varphi}(x,u^{(i)})-E_{\textrm{Surrogate}}[\varphi(x,U)]\right)^2$ |
| Standard deviation | $\mathbb{S}[\varphi(x,U)]$       | $S_{\textrm{PCE}}[\varphi(x,U)]=\sqrt{V_{\textrm{Surrogate}}[\varphi(x,U)]}$                                                                      |
| Margin             | $\textrm{Margin}[\varphi(x,U)]$  | $\textrm{Margin}_{\textrm{Surrogate}}[\varphi(x,U)]=E_{\textrm{Surrogate}}[\varphi(x,U)] + \kappa \cdot S_{\textrm{Surrogate}}[\varphi(x,U)]$     |
| Probability        | $\mathbb{P}[\varphi(x,U)\leq 0]$ | $P_{\textrm{PCE}}[\varphi(x,U)\leq 0]=E_N[\mathbb{1}_{\widehat{\varphi}(x,U)\leq 0}]$                                                             |
