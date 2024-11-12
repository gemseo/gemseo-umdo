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

The number of samples to build the surrogate model is mandatory
and must be set with the parameter `doe_n_samples`.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation="Surrogate",
    statistic_estimation_parameters={"doe_n_samples": 20}
)
```

## Options

### DOE algorithm

By default,
the formulation uses the [OpenTURNS](https://openturns.github.io)' DOE algorithm `OT_OPT_LHS`,
which is a Latin hypercube sampling (LHS) technique enhanced by simulated annealing,
a global optimization technique that
starts from an initial LHS
and improves it to maximize its discrepancy
and so to get a better space-filling LHS.

The DOE algorithm name can be set with the string option `doe_algo`
and its options with the dictionary parameter `doe_algo_options`.
When the DOE algorithm uses a random number generator,
the integer option `doe_seed` can be used for reproducibility purposes.

### Surrogate's options

This U-MDO formulation is based on a [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
By default,
this surrogate model is the [RBFRegressor][gemseo.mlearning.regression.algos.rbf.RBFRegressor] available in GEMSEO,
which wraps the [SciPy's RBF algorithm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html).
Use the `regressor_name` to change the kind of regressor
(use the name of a subclass of [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor])
and the `regressor_options` argument to set the options of the [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].
For example,
set `regressor_name` to `"LinearRegressor"` to use a linear regressor,
`regressor_options` to `{"function": "cubic"}` for a
[RBFRegressor][gemseo.mlearning.regression.algos.rbf.RBFRegressor] based on a cubic function
and `regressor_n_samples` to `100` to estimate the statistics with 100 Monte Carlo simulations instead of 10000.

!!! note "API"
    Use `statistic_estimation_parameters`
    to set the DOE algorithm and the surrogate's options,
    e.g.

    ``` py
    scenario = UMDOScenario(
        disciplines,
        mdo_formulation_name,
        objective_name,
        design_space,
        uncertain_space,
        statistic_name,
        statistic_estimation_parameters={
            "doe_algo": "OT_MONTE_CARLO",
            "doe_n_samples": 20,
            "doe_algo_options": {"n_processes": 2},
            "regressor_name": "PolynomialRegressor",
            "regressor_n_samples": 100,
            "regressor_options": {"degree": 3}
        }
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
for the expectation, the standard deviation, the variance, the margin and the probability,
by sampling the surrogate model $\widehat{\varphi}$ of $\varphi$.

| Statistic          | Notation                         | Estimator                                                                                                                                        |
|--------------------|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| Mean               | $\mathbb{E}[\varphi(x,U)]$       | $E_{\textrm{Surrogate}}[\varphi(x,U)]=\frac{1}{N}\sum_{i=1}^N\widehat{\varphi}(x,U^{(i)})$                                                       |
| Variance           | $\mathbb{V}[\varphi(x,U)]$       | $V_{\textrm{Surrogate}}[\varphi(x,U)]=\frac{1}{N-1}\sum_{i=1}^N\left(\widehat{\varphi}(x,U^{(i)})-E_{\textrm{Surrogate}}[\varphi(x,U)]\right)^2$ |
| Standard deviation | $\mathbb{S}[\varphi(x,U)]$       | $S_{\textrm{PCE}}[\varphi(x,U)]=\sqrt{V_{\textrm{Surrogate}}[\varphi(x,U)]}$                                                                     |
| Margin             | $\textrm{Margin}[\varphi(x,U)]$  | $\textrm{Margin}_{\textrm{Surrogate}}[\varphi(x,U)]=E_{\textrm{Surrogate}}[\varphi(x,U)] + \kappa \times S_{\textrm{Surrogate}}[\varphi(x,U)]$   |
| Probability        | $\mathbb{P}[\varphi(x,U)\leq 0]$ | $P_{\textrm{PCE}}[\varphi(x,U)\leq 0]=E_N[\mathbb{1}_{\widehat{\varphi}(x,U)\leq 0}]$                                                            |
