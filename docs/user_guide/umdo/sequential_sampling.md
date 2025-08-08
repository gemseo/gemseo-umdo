<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Sequential sampling

[SequentialSampling][gemseo_umdo.formulations.sequential_sampling.SequentialSampling]
is a U-MDO formulation that estimates the statistics unbiasedly
by using Monte Carlo sampling.
Contrary to [Sampling][gemseo_umdo.formulations.sampling.Sampling],
this U-MDO formulation does not use a constant sample size
but a sample size that increases with the iterations of the optimization loop.

The number of samples `n_samples` corresponds to the maximum number of samples
for a given iteration of the optimization loop.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation_settings=SequentialSampling_Settings(n_samples=20),
)
```

## Settings

### Algorithm

By default,
the formulation uses the algorithm `OT_OPT_LHS`
to get a good space-filling design of experiments (DOE),
with only 10 samples.

!!! question "DOE algorithms"
    Read the [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/doe.html#algorithms)
    for more information about the available DOE algorithms.

This maximum number of samples can be changed with the parameter `n_samples`
and the DOE algorithm name can be changed with the parameter `doe_algo_settings`,
which is a Pydantic model deriving from [BaseDOESettings][gemseo.algos.doe.base_doe_settings.BaseDOESettings].
When `n_samples` is `None` (default) and `doe_algo_settings` has a field `n_samples`,
then this field is considered as the maximum number of samples.
When `doe_algo_settings` has a field `seed` and its value is `None`,
then the U-MDO formulation will use [SEED][gemseo.utils.seeder.SEED].

!!! note "API"
    Use `statistic_estimation_settings`
    to set the DOE algorithm name and settings,
    e.g.

    ``` py
    settings = SequentialSampling_Settings(doe_algo_settings=OT_MONTE_CARLO(n_samples=20, n_processes=2))
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

### Sampling size profile

By default,
the number of samples is equal to 2 at the first iteration
and is incremented by 1 at each iteration of the optimization loop.

These values can be changed with the statistic estimation parameters
`initial_n_samples` and `n_samples_increment`.
In particular,
`n_samples_increment` can be either the increment of the sampling size
or a function computing this increment from the current sampling size.

## Statistics

This U-MDO formulation has been implemented
for the expectation, the standard deviation, the variance, the margin and the probability,
The estimators are given below at the $k$-th iteration of the optimization loop.

| Statistic          | Notation                         | Estimator                                                                                                      |
|--------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------|
| Mean               | $\mathbb{E}[\varphi(x,U)]$       | $E_{N_k}[\varphi(x,U)]=\frac{1}{N_k}\sum_{i=1}^{N_k}\varphi(x,u^{(i)})$                                        |
| Variance           | $\mathbb{V}[\varphi(x,U)]$       | $V_{N_k}[\varphi(x,U)]=\frac{1}{N_k-1}\sum_{i=1}^{N_k}\left(\varphi(x,u^{(i)})-E_{N_k}[\varphi(x,U)]\right)^2$ |
| Standard deviation | $\mathbb{S}[\varphi(x,U)]$       | $S_{N_k}[\varphi(x,U)]=\sqrt{V_{N_k}[\varphi(x,U)]}$                                                           |
| Margin             | $\textrm{Margin}[\varphi(x,U)]$  | $\textrm{Margin}_{N_k}[\varphi(x,U)]=E_{N_k}[\varphi(x,U)]+\kappa\cdot S_{N_k}[\varphi(x,U)]$                  |
| Probability        | $\mathbb{P}[\varphi(x,U)\leq 0]$ | $P_{N_k}[\varphi(x,U)\leq 0]=E_{N_k}[\mathbb{1}_{\varphi(x,U)\leq 0}]$                                         |
