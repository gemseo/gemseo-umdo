<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Sampling

[Sampling][gemseo_umdo.formulations.sampling.Sampling]
is a U-MDO formulation that estimates the statistics unbiasedly
using Monte Carlo sampling.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation_settings=Sampling_Settings(n_samples=20),
)
```

## Settings

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
and the DOE algorithm name can be changed with the parameter `doe_algo_settings`,
which is a Pydantic model deriving from [BaseDOESettings][gemseo.algos.doe.base_doe_settings.BaseDOESettings].
When `n_samples` is `None` (default) and `doe_algo_settings` has a field `n_samples`,
then this field is considered.
When `doe_algo_settings` has a field `seed` and its value is `None`,
then the U-MDO formulation will use [SEED][gemseo.utils.seeder.SEED].

!!! note "API"
    Use `statistic_estimation_settings`
    to set the DOE algorithm name and settings,
    e.g.

    ``` py
    settings = Sampling_Settings(doe_algo_settings=OT_MONTE_CARLO(n_samples=20, n_processes=2)
    scenario = UMDOScenario(
        disciplines,
        mdo_formulation_name,
        objective_name,
        design_space,
        uncertain_space,
        statistic_name,
        statistic_estimation_settings=settings),
    )
    ```

## Statistics

This U-MDO formulation has been implemented
for the expectation, the standard deviation, the variance, the margin and the probability.

| Statistic          | Notation                         | Estimator                                                                                        |
|--------------------|----------------------------------|--------------------------------------------------------------------------------------------------|
| Mean               | $\mathbb{E}[\varphi(x,U)]$       | $E_N[\varphi(x,U)]=\frac{1}{N}\sum_{i=1}^N\varphi(x,u^{(i)})$                                    |
| Variance           | $\mathbb{V}[\varphi(x,U)]$       | $V_N[\varphi(x,U)]=\frac{1}{N-1}\sum_{i=1}^N\left(\varphi(x,u^{(i)})-E_N[\varphi(x,U)]\right)^2$ |
| Standard deviation | $\mathbb{S}[\varphi(x,U)]$       | $S_N[\varphi(x,U)]=\sqrt{V_N[\varphi(x,U)]}$                                                     |
| Margin             | $\textrm{Margin}[\varphi(x,U)]$  | $\textrm{Margin}_N[\varphi(x,U)]=E_N[\varphi(x,U)]+\kappa\cdot S_N[\varphi(x,U)]$                |
| Probability        | $\mathbb{P}[\varphi(x,U)\leq 0]$ | $P_N[\varphi(x,U)\leq 0]=E_N[\mathbb{1}_{\varphi(x,U)\leq 0}]$                                   |

## Gradient-based optimization

When the multidisciplinary process is differentiable,
and a gradient-based optimizer is used,
analytical derivatives are implemented with the following statistics:
mean, standard deviation, variance and margin.
For probability statistics,
only derivatives approximated by finite differences are currently available.
