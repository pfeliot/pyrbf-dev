Radial Basis Functions (RBF)
****************************

The RBF model has the following form

.. math::
    :label: rbf
    
    \hat{f_c}(x) = \sum_{i=1}^m \lambda_i \phi_c( \lVert x - a_i \rVert_2) + t(x),
    
where the subscript :math:`c` indicates the dependence of the model to some parameter :math:`c`, the :math:`(a_i)_{1 \leq i \leq m}` 
are :math:`m` support points in :math:`\mathbb{R}^d` and :math:`\phi_c` and :math:`t` are respectively the model's kernel and the model's tail:

.. math::

    t(x) = 
    \left\{
    \begin{array}{ll}
    \beta^T x + \alpha & \text{Linear tail}, \\
    \alpha & \text{Constant tail}.
    \end{array}
    \right.

Interpolation formulas
----------------------

Given data points :math:`(x_i,\, y_i)_{1 \leq i, \leq n}`, the model :eq:`rbf` can be made interpolant by setting :math:`m = n` 
and :math:`a_i = x_i`, :math:`1 \leq i \leq n`, and by chosing the parameters :math:`(\lambda_i)_{1 \leq i \leq n}` and the 
tail's parameters :math:`\alpha` or :math:`(\alpha,\, \beta)` that solve a linear system of equations
of the form:

.. math::
    :label: system

    Ka = b

with

.. math::
    
    K = 
    \begin{pmatrix}
        \Phi_c & P \\
        P^T & 0
    \end{pmatrix},
    a = 
    \begin{pmatrix}
        \Lambda \\
        \Gamma
    \end{pmatrix},
    \text{ and } b =
    \begin{pmatrix}
        Y \\
        0
    \end{pmatrix},
    
where 

.. math::

    \left\{
    \begin{array}{lcl}
    \Phi_c &=& (\phi_c( \lVert x_i - x_j \rVert_2))_{1 \leq i,\,j \leq n}, \\
    \Lambda &=& (\lambda_1,\, \ldots,\, \lambda_n)^T, \\
    Y &=& (y_1,\, \ldots,\, y_n)^T,
    \end{array}
    \right.
    
and :math:`\Gamma = \alpha` and :math:`P = (1,\, \ldots,\, 1)^T` for a constant tail, or :math:`\Gamma = (\beta_1,\, \ldots,\, \beta_d, \alpha)^T` and

.. math::

    P = 
    \begin{pmatrix}
        x_1 & 1 \\
        \vdots & \vdots \\
        x_n & 1
    \end{pmatrix}
    
for a linear tail.

The system :eq:`system` can be solved efficiently using a PLU decomposition. 
    
Leave-One-Out (LOO) residuals
-----------------------------

To improve the generalization capabilities of the model, the kernel parameter :math:`c` can be chosen to minimize the sum of squared errors 
associated with a leave-one-out analysis (see, e.g., [1]_). 

Denote

.. math::
    :label: loo_rbf
    
    \hat{f_c}^{(k)}(x) = \sum_{i \neq k} \lambda_i^{(k)} \phi_c( \lVert x - x_i \rVert_2) + t^{(k)}(x),

the model obtained when :math:`(x_k,\, y_k)` is removed from the training set and write :math:`K^{(k)}`, :math:`a^{(k)}` and :math:`b^{(k)}` the associated 
system matrices:

.. math::
    :label: loo_system
    
    K^{(k)} a^{(k)} = b^{(k)}.

The leave-one-out residuals are defined as

.. math::

    \varepsilon_k = y_k - \hat{f_c}^{(k)}(x_k) \,, \quad 1 \leq k \leq n.
    
Observe that, if :math:`a \in \mathbb{R}^{n+l}`, where :math:`l = 1` for a constant tail and :math:`l = d+1` for a linear tail, is such that :math:`a_k = 0`, then

.. math::
    :label: prop

    Ka = z \quad \Rightarrow \quad K^{(k)} a^{(k)} = z^{(k)}.
    
Such a vector can be obtained by introducing the solution :math:`u^{[k]}` to the system

.. math::
    :label: identity
    
    Ku^{[k]} = e^{[k]},
    
where :math:`e^{[k]} = (0,\, \ldots,\, 0,\, 1,\, 0,\, \ldots,\, 0)^T` is the :math:`k`-th column of the :math:`(n+l) \times (n+l)` identity matrix,
and by defining

.. math::
    
    a^{[k]} = a - \frac{a_k}{u_k^{[k]}} u^{[k]}
    
where :math:`a` is solution to :eq:`system`. By design, :math:`a^{[k]}_k = 0` and

.. math::

    K a^{[k]} = Ka - \frac{a_k}{u_k^{[k]}} K u^{[k]} 
              = b - \frac{u_k}{a_k^{[k]}} e^{[k]} 
              = (y_1,\, \ldots,\, y_{k-1},\, y_k - \frac{a_k}{u_k^{[k]}},\, y_{k+1},\, \ldots,\, y_n,\, 0,\, \ldots,\, 0).
    
Thus, using :eq:`prop`, we obtain that the vector 
:math:`(a^{[k]}_1,\, \ldots,\, a^{[k]}_{k-1},\, a^{[k]}_{k+1},\, \ldots,\, a^{[k]}_{n+l})` is solution to :eq:`loo_system` and hence, this is :math:`a^{(k)}`.

Using this result,

.. math::
    
    \begin{array}{lcl}
    \hat{f_c}^{(k)}(x_k)
    &=& \sum_{i \neq k} \lambda_i^{(k)} \phi_c( \lVert x_k - x_i \rVert_2) + t^{(k)}(x_k) \\[5pt]
    &=& \sum_{i \neq k} a^{[k]}_i \phi_c( \lVert x_k - x_i \rVert_2) + t^{(k)}(x_k) \\[5pt]
    &=& \sum_{i = 1}^n a^{[k]}_i \phi_c( \lVert x_k - x_i \rVert_2) + t^{(k)}(x_k) \\[5pt]
    &=& (Ka^{[k]})_k \\[5pt]
    &=& y_k - \frac{a_k}{u_k^{[k]}}
    \end{array}
    
and thus

.. math::
    
    \varepsilon_k = \frac{a_k}{u_k^{[k]}} = \frac{\lambda_k}{K_{k,\,k}^{-1}}\,, \quad 1 \leq k \leq n

Prediction error
----------------

A measure of the RBF model prediction error has been proposed by [2]_. For a given :math:`\mathrm{x} \in \mathbb{R}^d`, the 
prediction error :math:`\sigma(\mathrm{x})` of the model at the point :math:`\mathrm{x}` can be computed 
using the following formula:

.. math::

    \sigma^2(\mathrm{x}) = \phi_c(0) - a_\mathrm{x}^T K^{-1} a_\mathrm{x},
    
where :

.. math::
    
    a_\mathrm{x} = 
    \begin{pmatrix}
        u_\mathrm{x} \\
        P_\mathrm{x}
    \end{pmatrix}
    
with :math:`u_\mathrm{x} = (\phi_c( \lVert \mathrm{x} - x_1 \rVert_2),\, \ldots,\, \phi_c( \lVert \mathrm{x} - x_n \rVert_2))^T` 
and :math:`P_\mathrm{x} = 1` for a constant tail or :math:`P_\mathrm{x} = (\mathrm{x}_0,\, \ldots,\, \mathrm{x}_d,\, 1)^T`
for a linear tail.

References
----------

.. [1] Rippa, S. (1999). *An algorithm for selecting a good value for the parameter c in radial basis function interpolation.* Advances in Computational Mathematics, 11(2-3), 193-210.
.. [2] Gutmann, H. M. (2001). *A radial basis function method for global optimization.* Journal of global optimization, 19(3), 201-227.