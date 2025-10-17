<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# MDO under uncertainty

## Introduction

### Optimization problem

A standard optimization problem aims to
find a vector $x^*$
minimizing an objective function $f$
over a search space $\mathcal{X}\subset\mathbb{R}^d$
while satisfying inequality constraints $g(x)\leq 0$
and equality constraints $h(x)=0$:

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}& & f(x) \\
&\operatorname{subject\;to}
& &g(x) \leq 0 \\
&&&h(x) = 0
\end{align}
$$

where $f:\mathcal{X}\to\mathbb{R}$,
$g:\mathcal{X}\to\mathbb{R}^{p_g}$
and $h:\mathcal{X}\to\mathbb{R}^{p_h}$.

Any optimization problem of the form

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}& & f_{\textrm{cost}}(x) \\
&\underset{x\in\mathcal{X}}{\operatorname{maximize}}&
& f_{\textrm{performance}}(x) \\
&\operatorname{subject\;to} & &g_n(x) \leq t_{g_n} \\
&&&g_p(x) \geq t_{g_p} \\
&&&\tilde{h}(x) = t_h
\end{align}
$$

can be reduced to such a standard optimization problem:

- an objective to minimize,
- upper inequality constraints with bounds equal to 0,
- equality constraints with right-hand sides equal to 0.

!!! note "API"

    In [GEMSEO](https://www.gemseo.org),
    the user instantiates an
    [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
    from a [DesignSpace][gemseo.algos.design_space.DesignSpace],
    defines its objective functions and constraints
    with
    [MDOFunction][gemseo.core.mdo_functions.mdo_function.MDOFunction] objects
    and solves it with an algorithm
    from a [BaseDriverLibrary][gemseo.algos.base_driver_library.BaseDriverLibrary].
    This algorithm can be either an optimizer or a design of experiments (DOE).

    ??? example

        The optimization problem

        $$
        \begin{align}
        &\underset{x\in[-1,1]}{\operatorname{minimize}}& & x^2 \\
        &\operatorname{subject\;to} & & x^3 \ge 0.1
        \end{align}
        $$

        can be solved with the Python lines

        ``` py
        from gemseo import execute_algo
        from gemseo.algos.optimization_problem import Optimization
        from gemseo.algos.design_space import DesignSpace
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        design_space = DesignSpace()
        design_space.add_variable("x", lower_bound=-1., upper_bound=1.)

        problem = OptimizationProblem(design_space)
        problem.objective = MDOFunction(lambda x: x**2, "f")
        problem.add_constraint(MDOFunction(lambda x: x**3, "g"), positive=True, value=0.1)

        execute_algo(problem, algo_name="PYDOE_FULLFACT", n_samples=10, algo_type="doe")
        ```


### Optimization problem under uncertainty

The models are often subject to uncertainties.
There are different ways of classifying and modeling uncertainties.
GEMSEO-UMDO uses the probability theory
to model both the aleatory and epistemic uncertainties as random variables.

Then,
the objective $f(x,U)$ and the constraints $g(x,U)$ and $h(x,U)$,
where $U$ denotes random inputs,
are in turn random variables
and the standard optimization problem is replaced by

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}& & \mathbb{K}_f[f(x,U)] \\
&\operatorname{subject\;to}
& &\mathbb{K}_g[g(x,U)] \leq 0 \\
&&&\mathbb{K}_h[h(x,U)] = 0
\end{align}
$$

where $\mathbb{K}_f$, $\mathbb{K}_g$ and $\mathbb{K}_h$ are statistic operators,
e.g.

- the expectation operator $\mathbb{E}$,
- the standard deviation operator $\mathbb{S}$,
- the variance operator $\mathbb{V}$,
- a margin, i.e. a combination of expectation and standard deviation operators parametrized by a weight $\kappa$: $\mathbb{E}+\kappa\cdot\mathbb{S}$,
- a probability operator $\mathbb{P}$ parametrized by bounds $m$ and $M$.

Concerning the margin,
the sign of $\kappa$ depends on the function type:

- $\kappa$ must be positive for an objective to minimize,
- $\kappa$ must be negative for an objective to maximize,
- $\kappa$ must be positive for a negativity constraint,
- $\kappa$ must be negative for a positivity constraint,
- $\kappa$ can be either positive of negative for an observable.

GEMSEO-UMDO will take care of changing the $\kappa$ sign according to these rules. No need to worry about it.

In practice,
the statistics $\mathbb{K}_f[f(x,U)]$, $\mathbb{K}_g[g(x,U)]$ and $\mathbb{K}_h[h(x,U)]$ are unknown
and the operators $\mathbb{K}_f$, $\mathbb{K}_g$ and $\mathbb{K}_h$ are replaced
by data-based operators $\widehat{\mathbb{K}}_f$, $\widehat{\mathbb{K}}_g$ and $\widehat{\mathbb{K}}_h$.
For example,
in the case of the expectation operator $\mathbb{E}$ and the Monte Carlo (MC) sampling technique,
the statistic $\mathbb{E}[f(x,U)]$ is replaced
by the statistic estimator $\widehat{\mathbb{E}}[f(x,U)]=\frac{1}{N}\sum_{i=1}^Nf(x,u^{(i)})$
where $u^{(1)},\ldots,u^{(N)}$ are $N$ independent realizations of $U$.

Typically,
a margin is applied to the objective to ensure a robust optimum $x^*$.
Indeed, minimizing margin $\mathbb{E}[f(x,U)]+\kappa\cdot\mathbb{S}[f(x,U)]$ pushes towards a compromise between

1. a small value of $f(x^*,u)$ by minimizing $\mathbb{E}[f(x,U)]$,
2. a small dispersion around $f(x^*,u)$ by minimizing $\mathbb{S}[f(x,U)]$.

For reliability purposes,
probabilities could be applied to inequality constraints,
e.g. $\mathbb{P}[g(x,U)\geq\epsilon]$ or $\mathbb{P}[g(x,u)>0]-\varepsilon$,
or equality constraints,
e.g. $\mathbb{P}[|h(x,U)|\geq\epsilon]$.

!!! note "Margin under normal assumption"

    When $\phi(x,U)$ is normally distributed,
    the margin $\mathbb{E}[\phi(x,U)]+q_\alpha\cdot\mathbb{S}[\phi(x,U)]$
    corresponds to the $\alpha$-quantile of $\phi(x,U)$
    where $q_\alpha$ is the $\alpha$-quantile
    of the standard Gaussian distribution.
    For that reason,
    2 or 3 are common candidates for $\kappa$
    as in this case,
    the margins correspond to
    the 0.975- and 0.999- quantiles of $\phi(x,U)$ respectively.

### Multidisciplinary optimization (MDO) problem

In complex systems,
the quantities $f(x)$, $g(x)$ and $h(x)$ are often computed by $M$ models,
called *disciplines*,
which can be weakly or strongly coupled.
The vector $x$ can be split into

- a sub-vector $x_0$ shared by at least two disciplines,
- sub-vectors $x_{1\ldots M}=\{x_1,\ldots,x_M\}$, where $x_i$ is specific to the $i$-th discipline.

Then,
the problem can be rewritten as the multidisciplinary optimization (MDO) problem

$$
\begin{align}
&\underset{x\in\mathcal{X},\,y\in\mathcal{Y}}{\operatorname{minimize}}&&f(x,y)\\
&\operatorname{subject\;to}
& &g(x,y) \leq 0 \\
&&&h(x,y) = 0 \\
&&&y=c(x,y)
\end{align}
$$

where $c_i:x_0,x_i,y_{-i}\mapsto y_i$ represents the $i$-th discipline
with $y_{-i}=\{y_j\}_{j\in\{1,\ldots,M\}\setminus\{i\}}$.

This MDO problem implies that
the optimum $(x^*,y^*)$ must be multidisciplinary feasible,
i.e. satisfying the coupling equations $y=c(x,y)$.
Solving a non-linear equation system is called a *multidisciplinary analysis* (MDA).

??? example "Example: MDA with linear disciplines"

    Let us consider a simple MDO problem with two linear disciplines given by
    $c_1: x_0,x_1,y_2 \mapsto x_0 + x_1 + y_2$
    and
    $c_2: x_0,x_2,y_1 \mapsto x_0 + x_2 + 2y_1$.
    Let us define the objective function as
    $f(x_0,x_1,x_2,y_1,y_2)=c_1(x_0,x_1,y_2)^2+c_2(x_0,x_2,y_1)^2$.
    The coupling equations are
    $c_1(x_0,x_1,y_2)=y_1$ and $c_2(x_0,x_2,y_1)=y_2$.
    In this linear case,
    they can be solved analytically:
    $y_1(x) = -2x_0-x_1-x_2$ and $y_2(x) = -3x_0-2x_1-x_2$.
    Then,
    the objective function output can be rewritten as a function of $x$ only:
    $f(x,y(x))=y_1(x)^2+y_2(x)^2$,
    and the MDO problem becomes a simple optimization problem.

In the case of non-linear disciplines,
the MDA can be solved with Newton's method or a fixed-point technique.

### MDO formulations

The efficient resolution of an MDO problem involves
finding a suitable rewriting of the problem,
called *MDO formulation* or *architecture*[@MartinsSurvey],
which is accompanied by a characteristic process (workflow and dataflow).
We present some MDO formulations just after,
with a very simple version under uncertainty
where the statistics are estimated by MC sampling using $N$ realizations of $U$.
More advanced estimation techniques can be found in GEMSEO-UMDO.

#### MDF

The MDF (multidisciplinary feasible) formulation of the MDO problem is certainly the best-known:

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}&&f(x,y(x))\\
&\operatorname{subject\;to}
& &g(x,y(x)) \leq 0 \\
&&&h(x,y(x)) = 0. \\
\end{align}
$$

This architecture performs an MDA at each design point $x$,
which, by the implicit function theorem, allows us to rewrite $y$ as $y(x)$, a function of $x$.
Because of this point-by-point coupling management,
the MDF formulation is qualified as *coupled*.

In presence of uncertainties,
we have:

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}&&\mathbb{K}_f[f(x,y(x,U),U)]\\
&\operatorname{subject\;to}
& &\mathbb{K}_g[g(x,y(x,U),U)] \leq 0 \\
&&&\mathbb{K}_h[h(x,y(x,U),U)] = 0. \\
\end{align}
$$

Here is a naive MC version:

$$
\begin{align}
&\underset{x\in\mathcal{X}}{\operatorname{minimize}}&&\widehat{\mathbb{K}}_f[f(x,y(x,U),U)]\\
&\operatorname{subject\;to}
& &\widehat{\mathbb{K}}_g[g(x,y(x,U),U)] \leq 0 \\
&&&\widehat{\mathbb{K}}_h[h(x,y(x,U),U)] = 0. \\
\end{align}
$$

where $\widehat{\mathbb{K}}_{\phi}[\phi(x,y(x,U),U)]$ is written
from the samples $\{\phi\!\left(x,y\!\left(x,u^{(i)}\right),u^{(i)}\right)\}_{i\in\{1,\ldots,N\}}$.

Note that the MDA is performed sample by sample,
thus multiplying the computational cost of an iteration of the optimization loop by $N$
with respect to the uncertainty-free case (without considering possible parallelizations).

#### IDF

The IDF (individual disciplinary feasible) formulation is also popular:

$$
\begin{align}
&\underset{x\in\mathcal{X},\,\tilde{y}\in\mathcal{Y}}{\operatorname{minimize}}&&f(x,c(x,\tilde{y}))\\
&\operatorname{subject\;to}
& &g(x,c(x,\tilde{y})) \leq 0 \\
&&&h(x,c(x,\tilde{y})) = 0 \\
&&&\tilde{y} = c(x,\tilde{y}).
\end{align}
$$

This architecture evaluates the disciplines independently
at each iteration of the optimization loop
and is thus qualified as *uncoupled*.
The multidisciplinary feasibility is ensured
at convergence of the optimization algorithm
by means of the consistency constraints $\tilde{y} = c(x,\tilde{y})$
and the additional optimization variables $\tilde{y}$.

In presence of uncertainties,
we have:

$$
\begin{align}
&\underset{x\in\mathcal{X},\,\tilde{Y}}{\operatorname{minimize}}&&\mathbb{K}_f[f(x,c(x,\tilde{Y},U),U)]\\
&\operatorname{subject\;to}
& &\mathbb{K}_g[g(x,c(x,\tilde{Y},U),U)] \leq 0 \\
&&&\mathbb{K}_h[h(x,c(x,\tilde{Y},U),U)] = 0 \\
&&&\tilde{Y} = c(x,\tilde{Y},U).
\end{align}
$$

where $\tilde{Y}$ is a random variable.

Here is a naive MC version:

$$
\begin{align}
&\underset{x\in\mathcal{X},\tilde{y}^{(1)},\ldots,\tilde{y}^{(N)}}{\operatorname{minimize}}&&\widehat{\mathbb{K}}_f[f(x,c(x,\tilde{Y},U),U]\\
&\operatorname{subject\;to}
& &\widehat{\mathbb{K}}_g[g(x,c(x,\tilde{Y},U),U] \leq 0 \\
&&&\widehat{\mathbb{K}}_h[h(x,c(x,\tilde{Y},U),U] = 0 \\
&&&\tilde{y}^{(1)} = c\!\left(x,\tilde{y}^{(1)},u^{(1)}\right) \\
&&&\vdots\\
&&&\tilde{y}^{(N)} = c\!\left(x,\tilde{y}^{(N)},u^{(N)}\right).
\end{align}
$$

where $\widehat{\mathbb{K}}_{\phi}[\phi(x,c(x,\tilde{Y},U),U)]$ is written
from the samples $\{\phi\!\left(x,c\!\left(x,\tilde{y}^{(i)},u^{(i)}\right),u^{(i)}\right)\}_{i\in\{1,\ldots,N\}}$.

Note that the dimension of the consistency constraints space
as well as the dimension of the additional optimization variables space
increases by a factor of $N$,
which can be very problematic,
especially when the dimension of the coupling variables space is large.

#### Bi-level

Bi-level formulations split the optimization problem
into a top-level optimization problem controlling the shared design variables $x_0$
and a sublevel optimization problem controlling the local design variables $x_{1\ldots M}$.

In particular,
Gazaix *et al.* (2019)[@Gazaix2019] proposed to
rewrite the initial MDO problem into a sequence of bi-level optimization problems
whose sublevels are split into independent disciplinary optimization problems.
This decomposition according to the disciplines makes
the reuse of existing disciplinary optimization processes possible
as well as their execution in parallel.
It can also be done
according to the nature of the design variables (continuous, discrete or mixed variables, etc.)
and whether certain derivatives are available,
to take advantage of specialized optimizers,
although in the following we present the disciplinary version.

Given the initial shared design values $x_0^{(0)}$ and the initial local design values $x_{1\ldots M}^{(0)}$,
the top-level optimization problem of the $k$-th term of this bi-level optimization problem sequence can be written as

$$
\begin{align}
&\underset{x_0\in\mathcal{X}_0}{\operatorname{minimize}}&&f\!\left(x_0, x_{1\ldots M}^{(k)}, y\!\left(x_0, x_{1\ldots M}^{(k)}\right)\right)\\
&\operatorname{subject\;to}
& &g\!\left(x_0, x_{1\ldots M}^{(k)}, y\!\left(x_0, x_{1\ldots M}^{(k)}\right)\right)\leq 0
\end{align}
$$

where $x_i^{(k)}$ is a solution of the disciplinary optimization problem

$$
\begin{align}
&\underset{x_i\in\mathcal{X}_i}{\operatorname{minimize}}&&f\!\left(x_0, x_i, x_{-i}^{(k-1)}, c\!\left(x_0,x_i,x_{-i}^{(k-1)},y\!\left(x_0,x_{1\ldots M}^{(k-1)}\right)\right)\right)\\
&\operatorname{subject\;to}
& &g_i\!\left(x_0, x_i, x_{-i}^{(k-1)}, c\!\left(x_0,x_i,x_{-i}^{(k-1)},y\!\left(x_0,x_{1\ldots M}^{(k-1)}\right)\right)\right)\leq 0
\end{align}
$$

with $x_{-i}^{(k-1)}=\left(x_j^{(k-1)}\right)_{j\in\{1,\ldots,M\}\setminus\{i\}}$
and $y\!\left(x_0,x_{1\ldots M}^{(k-1)}\right)$ the MDA solution at the design point $\left(x_0,x_{1\ldots M}^{(k-1)}\right)$.

In presence of uncertainties,
Aziz Alaoui (2025)[@AzizAlaoui2025] proposed a similar bi-level formulation.
The main problem can be written as

$$
\begin{align}
&\underset{x_0\in\mathcal{X}_0}{\operatorname{minimize}}&&\mathbb{K}_f\!\left[f\!\left(x_0, X_{1\ldots M}^{(k)}, y\!\left(x_0, X_{1\ldots M}^{(k)}, U\right), U\right)\right]\\
&\operatorname{subject\;to}
& & \mathbb{K}_g\!\left[g\!\left(x_0, X_{1\ldots M}^{(k)}, y\!\left(x_0, X_{1\ldots M}^{(k)}, U\right), U\right)\right]\leq 0
\end{align}
$$

where $X_i^{(k)}=x_i^{(k)}(U)$ is a random variable
such that for any realization $u$ of $U$,
$x_i^{(k)}(u)$ is a solution of the disciplinary optimization problem

$$
\begin{align}
&\underset{x_i\in\mathcal{X}_i}{\operatorname{minimize}}&&f\!\left(x_0, x_i, x_{-i}^{(k-1)}(u), c\!\left(x_0,x_i,x_{-i}^{(k-1)}(u),y\!\left(x_0,x_{1\ldots M}^{(k-1)}(u),u\right),u\right),u\right)\\
&\operatorname{subject\;to}
& &g_i\!\left(x_0, x_i, x_{-i}^{(k-1)}(u), c\!\left(x_0,x_i,x_{-i}^{(k-1)}(u),y\!\left(x_0,x_{1\ldots M}^{(k-1)}(u),u\right),u\right),u\right)\leq 0
\end{align}
$$

!!! note "Different in kind from MDF"
    Unlike its uncertainty-free version,
    which aims to be a reformulation of the MDF-formulated MDO problem,
    this bi-level formulation under uncertainty and its MDF counterpart are quite different in nature.
    While MDF is looking for an optimum design $x^*=(x_0^*,x_{1\ldots M}^*)$,
    this bi-level formulation is looking for a random optimum $X^*=(x_0^*,X_{1\ldots M}^*)$
    where only the shared design variables $x_0^*$ are fixed.
    The other design variables $X_{1\ldots M}^*$ are function of both this $x_0^*$ and the uncertain sources $U$
    and define the optimal choices possible according to the values of $U$.
    The idea behind this formulation is that
    because optimization under uncertainty leads to a solution more conservative than in the absence of uncertainties,
    it may be appropriate
    to fix only the global variables now
    and take time to fix the local variables,
    in the hope that the uncertainty $U$ will diminish over the course of the design process
    and so lead to less conservative solutions,
    i.e. a better objective value.

Here is a naive MC version:

$$
\begin{align}
&\underset{x_0\in\mathcal{X}_0}{\operatorname{minimize}}&&\widehat{K}_f\!\left[f\!\left(x_0, X_{1\ldots M}^{(k)}, y\!\left(x_0, X_{1\ldots M}^{(k)}, U\right), U\right)\right]\\
&\operatorname{subject\;to}
& &  \widehat{K}_g\!\left[g\!\left(x_0, X_{1\ldots M}^{(k)}, y\!\left(x_0, X_{1\ldots M}^{(k)}, U\right), U\right)\right]\leq 0
\end{align}
$$

where $\widehat{K}_{\phi}\!\left[\phi\!\left(x_0, X_{1\ldots M}^{(k)}, y\!\left(x_0, X_{1\ldots M}^{(k)}, U\right), U\right)\right]$
is written from the samples
$\left\{\phi\!\left(x_0, x_{1\ldots M}^{(k)}\!\left(u^{(i)}\right), y\!\left(x_0, x_{1\ldots M}^{(k)}\!\left(u^{(i)}\right), u^{(i)}\right)\right)\right\}_{i\in\{1,\ldots,N\}}$

and where $x_j^{(k)}(u^{(i)})$ is solution of

$$
\begin{align}
&\underset{x_i\in\mathcal{X}_i}{\operatorname{minimize}}&&f\!\left(x_0, x_i, x_{-i}^{(k-1)}\!\left(u^{(i)}\right), c\!\left(x_0,x_i,x_{-i}^{(k-1)}\!\left(u^{(i)}\right),y\!\left(x_0,x_{1\ldots M}^{(k-1)}\!\left(u^{(i)}\right),u^{(i)}\right),u^{(i)}\right),u^{(i)}\right)\\
&\operatorname{subject\;to}
& &g_i\!\left(x_0, x_i, x_{-i}^{(k-1)}\!\left(u^{(i)}\right), c\!\left(x_0,x_i,x_{-i}^{(k-1)}\!\left(u^{(i)}\right),y\!\left(x_0,x_{1\ldots M}^{(k-1)}\!\left(u^{(i)}\right),u^{(i)}\right),u^{(i)}\right),u^{(i)}\right)\leq 0
\end{align}
$$

## API

Here is an outline of the API to define and solve an MDO problem under uncertainty.
The key ingredients for the definition are
the disciplines,
the design space,
the constraints,
the objective(s) and
the U-MDO formulation combining an MDO formulation and a statistic estimation technique.
The MDO problem can be solved using either an optimizer or a DOE.
Go to the [examples](../../generated/examples/umdo/index.md) for more information.
In this section,
we present what is different compared to MDO without uncertainties

### Disciplines

When defining disciplines,
do not forget to declare the uncertain variables as input variables
so that the
[UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
and
[UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
can take them into account.

!!! example

    Let us implement an [Discipline][gemseo.core.discipline.discipline.Discipline]
    outputting $f(x,U)=(x_1+U)^2+(x_2+U)^2$:
    ``` py
    from numpy import array
    from gemseo.core.discipline.discipline import Discipline


    class MyDiscipline(Discipline):

        def __init__(self):
            super().__init__()
            self.input_grammar.update_from_names(["x1", "x2", "U"])
            self.default_input_data = {"x1": array([0.]), "x2": array([0.]), "U": array([0.5])}

        def _run(self, input_data):
            x1 = self.io.data["x1"]
            x2 = self.io.data["x2"]
            U = self.io.data["U"]
            y = (x1+U)**2 + (x2+U)**2
            self.io.update_output_data({"y": y})
    ```

    This discipline can be executed
    with different values of the uncertain variable $U$:
    ``` py
    discipline.execute()  # default value, i.e. U=0.5
    discipline.execute({"U": array([0.2])})  # custom value: U=0.2
    ```

### Uncertain space

The uncertain variables have to be defined
in a [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
with the method [add_random_variable][gemseo.algos.parameter_space.ParameterSpace.add_random_variable].

!!! example

    In the previous example,
    we could model the uncertain variable $U$ as a random variable
    triangularly distributed
    between 0.2 and 0.7 with a mode of 0.4:

    ``` py
    from gemseo.algos.parameter_space import ParameterSpace

    uncertain_space = ParameterSpace()
    uncertan_space.add_random_variable(
        "U", "OTTriangularDistribution", minimum=0.2, maximum=0.7, mode=0.4
    )
    ```

### Scenario

Given these disciplines and uncertain space,
as well as a design space, objective(s) and constraint(s),
the MDO problem under uncertainty can be set up by

- [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
  to solve it using a DOE,
- [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
  to solve it using an optimizer or a DOE.

You also need to fill in the statistics associated with the objective(s) and constraint(s),
and the U-MDO formulation, combining an statistics estimation technique and an MDO formulation.

!!! note "API"

    The API of [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
    is deliberately similar to
    the API of [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario].
    And the same for
    [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
    and
    [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario].
    This choice was made not only to simplify the user's life,
    but also because an MDO problem under uncertainty
    is first and foremost an MDO problem.

!!! example

    Continuing the previous example,
    we seek to minimize $\mathbf{E}[(x_1+U)^2+(x_2+U)^2]$
    over the domain $[-1,1]^2$
    with the gradient-free optimization algorithm COBYLA
    and a Monte Carlo estimator of the expectation.

    ``` py
    from gemseo.algos.design_space import DesignSpace
    from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
    from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario

    design_space = DesignSpace()
    design_space.add_variable("x1", lower_bound=-1., upper_bound=1.)
    design_space.add_variable("x2", lower_bound=-1, upper_bound=1.)

    scenario = UMDOScenario(
        [discipline],
        "DisciplinaryOpt",
        "y",
        design_space,
        uncertain_space,
        "Mean",
        statistic_estimation_settings=Sampling_Settings(n_samples=100),
    )
    scenario.execute(algo_name="NLOPT_COBYLA", max_iter=50)
    ```

### Statistic estimation

[UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]
and [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
can use [Sampling_Settings][gemseo_umdo.formulations.sampling_settings.Sampling_Settings],
with `n_samples` set to a certain value $N$,
to estimate the statistics associated with $f(x,U)$, $g(x,U)$ and $h(x,U)$ by [sampling](sampling.md),:

$$\!\left(f\!\left(x,u^{(i)}\right),g\!\left(x,u^{(i)}\right),h\!\left(x,u^{(i)}\right)\right)_{1\leq i \leq N}.$$

However,
when costly disciplines are involved,
this approach may be too expensive.
The rest of the [_MDO under uncertainty_](index.md) section of the user guide presents
other statistic estimation techniques to reduce the number of discipline evaluations,
such as
[control variates](control_variate.md),
[Taylor polynomials](taylor_polynomial.md)
and
[polynomial chaos expansions](pce.md).
Choosing an estimation technique and its options is made via the argument `statistic_estimation_settings`
of [UMDOScenario][gemseo_umdo.scenarios.umdo_scenario.UMDOScenario]
(or [UDOEScenario][gemseo_umdo.scenarios.udoe_scenario.UDOEScenario]),
which is a Pydantic model of settings,
e.g.
[ControlVariate_Settings][gemseo_umdo.formulations.control_variate_settings.ControlVariate_Settings],
[TaylorPolynomial_Settings][gemseo_umdo.formulations.taylor_polynomial_settings.TaylorPolynomial_Settings]
or [PCE_Settings][gemseo_umdo.formulations.pce_settings.PCE_Settings].

As of now,
only the
[``Sampling``](sampling.md),
[``SequentialSampling``](sequential_sampling.md)
and [``PCE``](sampling.md) U-MDO formulations
provide analytical derivatives of the statistics
when the disciplines and the multidisciplinary process generated by the MDO formulation are differentiable.
Therefore,
the other U-MDO formulations require gradient approximation to use gradient-based optimizers,
what can be expensive according to the dimension of the design space.

Finally,
it is important to bear in mind that
all these techniques make approximations in order to estimate statistics more cheaply than sampling.
Estimates are therefore made with a certain degree of precision,
and potentially with a certain degree of bias.
So,
at the end of an optimization process with a given statistic estimation technique T,
it is advisable to copy and paste the scenario
and replay it with [Sampling_Settings][gemseo_umdo.formulations.sampling_settings.Sampling_Settings]
parameterized by a non-negligible `n_samples`
at the design solution point `x_opt` found with T
(i.e. `scenario.execute(algo_name="CustomDOE", samples=at_least2d(x_opt))`)
to obtain a better estimate of the statistics at this point.
This avoids pitfalls,
such as concluding that a constraint is satisfied when it is not.
Note that this validation approach is also recommended in surrogate-based optimization (with or without uncertainty),
where it is advisable to
evaluate the objective and constraints with the original models at the end of an optimization loop,
due to the error of the surrogate models.

??? info "Implementation"

    Given a
    [DesignSpace][gemseo.algos.design_space.DesignSpace]
    and a collection of
    [Disciplines][gemseo.core.discipline.discipline.Discipline],
    a [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario]
    generates and solves an
    [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
    that corresponds to a
    [BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation].
    The resolution consists in sampling the objective and constraints
    over the [DesignSpace][gemseo.algos.design_space.DesignSpace],
    i.e. $(x^{(i)},f(x^{(i)},U),g(x^{(i)},U),h(x^{(i)},U))_{1\leq i \leq N}$,
    and returning either the $x^*$ minimizing $f$ while satisying $g$ and $h$
    or the $x^*$ that violates the least $g$ and $h$.

    GEMSEO-UMDO uses this sampling mechanism a first time
    with a [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
    instead of the [DesignSpace][gemseo.algos.design_space.DesignSpace]
    to estimate the statistics
    $\mathbb{K}_f[f(x,U)]$, $\mathbb{K}_g[g(x,U)]$ and $\mathbb{K}_h[h(x,U)]$
    based on the samples
    $(u^{(i)},f(x,u^{(i)}),g(x,u^{(i)}),h(x,u^{(i)}))_{1\leq i \leq N}$.
    In the case of Monte Carlo sampling,
    $N$ is the number of samples
    while in the case of Taylor expansion,
    $M\in\{1,1+q\}$ where $q$ is the dimension of the uncertain space
    depending on whether the derivatives are known
    or to be estimated by finite differences.
    These statistics estimators
    $\hat{\mathbb{K}}_f[f(x,U)]$,
    $\hat{\mathbb{K}}_g[g(x,U)]$ and
    $\hat{\mathbb{K}}_h[h(x,U)]$
    are then used to build a new
    [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]
    over the [DesignSpace][gemseo.algos.design_space.DesignSpace]:

    $$
    \begin{align}
    &\underset{x\in\mathcal{X}}{\operatorname{minimize}}
    & & \hat{\mathbb{K}}_f[f(x,U)] \\
    &\operatorname{subject\;to}
    & &\hat{\mathbb{K}}_g[g(x,U)] \leq 0 \\
    &&&\hat{\mathbb{K}}_h[h(x,U)] = 0.
    \end{align}
    $$

    Thus implemented,
    GEMSEO-UMDO should be able
    to set up any MDO problem under uncertainty
    from any [BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation]
    and any statistic estimation technique.
    This vision may be theoretical at the moment,
    but the ambition of GEMSEO-UMDO is to be
    an engine generating U-MDO formulations
    based on any MDO formulation and any statistic estimators.
