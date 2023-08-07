<!---
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
--->

# Taylor polynomial

The U-MDO formulation [TaylorPolynomial]
[gemseo_umdo.formulations.taylor_polynomial.TaylorPolynomial]
can solve an MDO problem
associated with an [MDOFormulation][gemseo.core.formulation.MDOFormulation]
with Taylor polynomials.

The Taylor polynomials are centered at $\mu=\mathbb{E}[U]$
where $U$ is the random input vector.

When the derivatives with respect to the uncertain variables are available,
this U-MDO formulation introduces no additional calculation cost
associated with taking the uncertainties into account.
Otherwise,
finite differences are computed
and so the additional cost is $d+1$ evaluations of the process
associated with the [MDOFormulation][gemseo.core.formulation.MDOFormulation]
where $d$ is the dimension of the uncertain space.

This U-MDO formulation has no mandatory parameters.

Here is a typical scenario template:

``` py
scenario = UMDOScenario(
    disciplines,
    mdo_formulation_name,
    objective_name,
    design_space,
    uncertain_space,
    statistic_name,
    statistic_estimation="TaylorPolynomial",
)
```

## Options

### Derivatives calculation

When the derivatives with respect to the uncertain variables are missing
or when the process
resulting from the [MDOFormulation][gemseo.core.formulation.MDOFormulation]
cannot be differentiated with respect to these variables,
this U-MDO formulation uses finite difference approximations.
One can also force the use of finite difference approximations
by setting the statistic estimation parameter `differentiation_method`
to `"finite_differences"`.

!!! note "API"
    Use `statistic_estimation_parameters` to set the options,
    e.g.

    ``` py
    scenario = UMDOScenario(
        disciplines,
        mdo_formulation_name,
        objective_name,
        design_space,
        uncertain_space,
        statistic_name,
        statistic_estimation="TaylorPolynomial",
        statistic_estimation_parameters={"differentiation_method": "finite_differences"}
    )
    ```

### Second-order

By default,
this U-MDO formulation uses first-order Taylor polynomials.
Second-order Taylor polynomials can also be used
by setting the statistic estimation parameter `second_order` to `True`.

!!! warning "Computational cost"
    As GEMSEO does not support second-order derivatives,
    the second-order derivatives are estimated by finite-differences
    from the first-order derivatives.
    When the dimension of the uncertain space is large
    or when the first-order derivatives
    are already finite difference approximations,
    using second-order Taylor polynomials can be very costly.

## Statistics

This formulation has been implemented for the expectation and variance,
as well as combinations of these statistics.

Here are the expressions when using first-order Taylor polynomials.

### Mean

$$\mathbb{E}[\varphi(x,U)]
\approx E_{\textrm{TP}_1}[\varphi(x,U)]
=\varphi(x,\mu)$$

### Variance

$$\mathbb{V}[\varphi(x,U)]
\approx V_{\textrm{TP}_1}[\varphi(x,U)]
=\nabla\varphi(x,\mu)^T\Sigma \nabla\varphi(x,\mu)$$

where
$\Sigma=\left(\textrm{cov}(U_i,U_j)\right)_{1\leq i,j\leq d}$
is the covariance matrix of $U$
and
$\nabla\varphi(x,\mu)=
\left(\frac{\partial\varphi(x,\mu)}{\partial u_i}\right)_{1\leq i \leq d}$
is the column-vector of the partial derivatives of $\varphi$
with respect to the uncertain variables.

### Standard deviation

$$\mathbb{S}[\varphi(x,U)]
\approx S_{\textrm{TP}_1}[\varphi(x,U)]
=\sqrt{V_{\textrm{TP}_1}[\varphi(x,U)]}$$

### Margin

$$\textrm{Margin}[\varphi(x,U)]
\approx \textrm{Margin}_{\textrm{TP}_1}[\varphi(x,U)]
=E_{\textrm{TP}_1}[\varphi(x,U)]+\kappa\times S_{\textrm{TP}_1}[\varphi(x,U)]$$
