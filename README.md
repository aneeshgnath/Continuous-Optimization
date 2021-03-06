# CONTINUOUS OPTIMIZATION #

## June 19, 2018 ##

In these days I have noticed that, sometimes the 1-dimensional research method plays a vital important role in the performance of solving the non-linear optimization problem, and obiviously it does not meet the "demand" if there is only `bisection method`, we are supposed to find more.

Moreover, for methods used for solving unconstrained optimization problems(`gradient descent method`, `newton method`, `conjugated direction method`, `quasi-newton method` and their variants), they all have used the information of the first order derivative of objective function! Sometimes the algorithm even requires the objective function to be **continuous differentiable**, which is a pretty ideal property!

I mean, if an objective function is already continuous differentiable, and the feasible region is the whole domain, we can get the analytic optimal solution quiet easily when the scale of problem is not huge and the objective function is not such weird.

In the lastest lab exam of the course Optimization the professor introduced such an optimization method which does not utilize the derivative, and it also have the global convergence property. This is a happy idea, because usually the optimization problems which we meet are without good properties.

However, I don't know if it is a heuristic algorithm, and I didn't check its theoretical foundation, which cause my unconvinced attitude to it X) Anyway, I will study it during the summer vacation, and provide the code at that time.

## June 6, 2018 ##

I will re-write this repository using OOP, also the theory part will be revised because it was not clear enough.

This repository will mainly dedicate on the **continuous** optimazition, including constrained ones and unconstrained ones. The **continuous** optimization is opposed to the **discrete** optimization, for example the integer optimization, for which we should search the optimum on a discrete feasible region.

We might be able to say continuous optimization is somehow easier to be solved than the discrete optimization. However, it's interesting to notice that in the discrete optimization problem, the fesible region (a set) is usually countable (In fact, even better because it is with finitely many elements). But in the continuous optimization, the fesible region is uncountable! Therefore a so-called contradiction rises: How could we solve the optimization problem in a much worse situation?

I think it's the **continuity of the object function** who helps us. That means, we can cultivate the derivative of object function, and, fortunately, utilize this information in the "continuous" feasible region case, but "discrete" we cannot.

Moreover, you may say that the object function might not be differentiable at every point, i.e. we cannot compute the derivative everywhere! Yes, you are right, but notice that at least usually we optimize the function  which is differentiable [almost everywhere](https://en.wikipedia.org/wiki/Almost_everywhere). Just think about the optimization on the [Dirichlet function](http://mathworld.wolfram.com/DirichletFunction.html). Although we have a continuous feasible region, I would say there is no diffenrence between continuous optimization and discrete optimization.

## April 14, 2017 ##

I'm sorry recently I do not have enough time to update this repository. In the middle of June the Linear Programming Model as the fundamental of the Zoutendijk Method, may be added in.

Regrads,

Zheng Li

## March 21, 2017 ##

**CATALOG:**

Note: You can read the principles in `README.pdf`

1. BASIC ALGORITHMS
- Numerically Computing Gradient Vector
- Numerically Computing Hessian Matrix
- Golden Section Method

2. OPTIMIZATION ALGORITHMS
- Unconstrained Optimization Algorithms
  - Gradient Descent Method
  - Guarded Newton Method
  - Conjugate Gradient Method (Using Fletcher-Reeves Method)
  - Quasi Newton Method (COMING SOON)
- Constrained Optimization Algorithms
  - Zoutendijk Method (COMING SOON)
  - Interior Point Method (COMING SOON)