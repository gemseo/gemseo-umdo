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

This is the default U-MDO formulation.
So, the argument `statistic_estimation` does not have to be set to use it.
However,
the number of samples is mandatory
and must be set with the parameter `n_samples`.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation_parameters={"n_samples": n_samples}
)
```

## Options

By default,
the formulation uses the DOE algorithm `OT_OPT_LHS`:
the Latin hypercube sampling (LHS)
enhanced by simulated annealing
of OpenTURNS.
Simulated annealing is a global optimization technique that
starts from an initial LHS
and improves it to maximize its discrepancy
and so to get a better space-filling LHS.

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

## Statistics

This U-MDO formulation has been implemented for the expectation, variance and probability,
as well as combinations of these statistics.

| Statistic          | Notation                         | Estimator                                                                                                                |
|--------------------|----------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| Mean               | $\mathbb{E}[\varphi(x,U)]$       | $E_N[\varphi(x,U)]=\frac{1}{N}\sum_{i=1}^N\varphi(x,U^{(i)})$                                                            |
| Variance           | $\mathbb{V}[\varphi(x,U)]$       | $V_N[\varphi(x,U)]=\frac{1}{N-1}\sum_{i=1}^N\left(\varphi(x,U^{(i)})-E_N[\varphi(x,U)]\right)^2$ |
| Standard deviation | $\mathbb{S}[\varphi(x,U)]$       | $S_N[\varphi(x,U)]=\sqrt{V_N[\varphi(x,U)]}$                                                                |
| Margin             | $\textrm{Margin}[\varphi(x,U)]$  | $\textrm{Margin}_N[\varphi(x,U)]=E_N[\varphi(x,U)]+\kappa\times S_N[\varphi(x,U)]$                                       |
| Probability        | $\mathbb{P}[\varphi(x,U)\leq 0]$ | $P_N[\varphi(x,U)\leq 0]=E_N[\mathbb{1}_{\varphi(x,U)\leq 0}]$                                                           |
