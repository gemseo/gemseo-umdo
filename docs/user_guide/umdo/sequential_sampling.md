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

The number of samples is mandatory
and must be set with the parameter `n_samples`.
It corresponds to the maximum number of samples
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
    statistic_estimation="SequentialSampling",
    statistic_estimation_parameters={"n_samples": n_samples}
)
```

## Options

### Algorithm

By default,
the formulation uses the algorithm `OT_OPT_LHS`
to get a good space-filling design of experiments (DOE).

!!! question "DOE algorithms"
    Read the [GEMSEO documentation](https://gemseo.readthedocs.io/en/stable/doe.html#algorithms)
    for more information about the available DOE algorithms.

The DOE algorithm name can be set with the string parameter `algo`
and its options with the dictionary parameter `algo_options`.

!!! note "API"
    Use `statistic_estimation_parameters`
    to set the algorithm name and parameters,
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
            "algo": "OT_MONTE_CARLO",
            "n_samples": 20,
            "algo_options": {"n_processes": 2}
        }
    )
    ```

### Sampling size profile

By default,
the number of samples is equal to 1 at the first iteration
and is incremented by 1 at each iteration of the optimization loop.

These values can be changed with the statistic estimation parameters
`initial_n_samples` and `n_samples_increment`.

## Statistics

This U-MDO formulation has been implemented for the expectation, variance and probability,
as well as combinations of these statistics.
The estimators are given below at the $k$-th iteration of the optimization loop.

| Statistic          | Notation                         | Estimator                                                                                                      |
|--------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------|
| Mean               | $\mathbb{E}[\varphi(x,U)]$       | $E_{N_k}[\varphi(x,U)]=\frac{1}{N_k}\sum_{i=1}^{N_k}\varphi(x,U^{(i)})$                                        |
| Variance           | $\mathbb{V}[\varphi(x,U)]$       | $V_{N_k}[\varphi(x,U)]=\frac{1}{N_k-1}\sum_{i=1}^{N_k}\left(\varphi(x,U^{(i)})-E_{N_k}[\varphi(x,U)]\right)^2$ |
| Standard deviation | $\mathbb{S}[\varphi(x,U)]$       | $S_{N_k}[\varphi(x,U)]=\sqrt{V_{N_k}[\varphi(x,U)]}$                                                           |
| Margin             | $\textrm{Margin}[\varphi(x,U)]$  | $\textrm{Margin}_{N_k}[\varphi(x,U)]=E_{N_k}[\varphi(x,U)]+\kappa\times S_{N_k}[\varphi(x,U)]$                 |
| Probability        | $\mathbb{P}[\varphi(x,U)\leq 0]$ | $P_{N_k}[\varphi(x,U)\leq 0]=E_{N_k}[\mathbb{1}_{\varphi(x,U)\leq 0}]$                                         |
