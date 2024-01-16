import numpy as np
from dataclasses import dataclass
from enum import Enum


class ConstType(Enum):
    DROP = 1
    QR = 2
    DIFF = 3

@dataclass
class Constraint:
  """
  Constraint storage. ``Z``, either holds the Qr-based correction matrix that needs
  to be multiplied with X, S, and D to make terms subject to the conventional sum-to-zero constraints
  applied also in mgcv (Wood, 2017), the column/row that should be dropped from those - then X can also no longer take
  on a constant, or ``None`` indicating that the model should be "difference re-coded" to enable sparse sum-to-zero
  constraints. The latter two are available in mgcv's ``smoothCon`` function by setting the ``sparse.cons`` argument to 1 or
  2 respectively.

  The QR-based approach is described in detail by Wood (2017) and is similar to just mean centering every basis function involved in the
  smooth and then dropping one column from the corresponding centered model matrix. The column-dropping approach is self-explanatory. The
  difference re-coding deserves some explanation. Consider the model:

  y = f(x)
    = b_1*f_1(x) + b_2*f_2(x) + b_3*f_3(x) + b_4*f_4(x) + b_5*f_5(x)
  
  In ``mssm`` f(x) will be parameterized via five B-spline basis functions f_i(x), each weighted by a coefficient b_i. Each of the f_i(x) has
  support only over a small range of x - resulting in a sparse model matrix X. This desirable property is lost when enforcing the conventional
  sum-to-zero constraints (either QR-based or by subtracting the mean of each basis over x from the corresponding f_i(x)). To maintain sparsity,
  one can enforce a coefficients-sum-to-zero constraint (Wood, 2020) by re-coding instead:

  Set b_1 = -b_2
  Re-write y = -b_2 * f_1(x) + b_2 * f_2(x) + b_3 * f_3(x) - b_3 * f_2(x) + b_4 * f_4(x) - b_4 * f_3(x) + b_5 * f_5(x) - b_5 * f_4(x)
             = -b_2 * f_1(x) + (b_2 - b_3) * f_2(x) + (b_3 - b_4) * f_3(x) + (b_4 - b_5) * f_4(x) + b_5 * f_5(x)
             = b_2 * (f_2(x) - f_1(x)) + b_3 * (f_3(x) - f_2(x)) + b_4 * (f_4(x) - f_3(x)) + b_5 * (f_5(x) - f_4(x))

  Line 3 shows how the constraint can be absorbed for model fitting by first computing new bases (f_i(x) - f_{i-1}(x)) and then estimating
  the coefficients based on those (this is implemented in mgcv's ``smoothCon`` function when setting ``sparse.cons=2``). Note that the constrained
  model, just like when using the QR-based or dropping approach, requires dropping one of the k original coefficients for estimation since only k-1
  coefficients are allowed to vary. The same sum-to-zero constraint can be achieved by fixing one of the central bases in the original model to it's neighbor (e.g., setting b_2 = -b_3) or
  by setting b_1= -b_2 - b_3 - b_4 - b_5. mssm fixes one of the central bases to it's neighbor.

  With a B-splines basis, it would be necessary that b_1 = b_2 = b_3 = b_4 = b_5 for the model to fit a constant f(x) over all values of x (Eilers & Marx, 1996).
  In the constrained model this is no longer possible due to b_1 = -b_2, effectively removing the constant from the space of functions that f(x) can approximate.
  Instead, in the constrained model if b_2 = b_3 = b_4 = b_5, f(x) summed over all x will be equal to zero!
  
  Importantly, the "new" bases obtained by computing (f_i(x) - f_{i-1}(x)) are all still relatively sparse. This is because neighboring
  f_i(x) overlap in their support and because the support of each individual f_i(x) is narrow, i.e., f_i(x) is sparse itself.

  However, this is not a true centering constraint: f(x) will not necessarily be orthogonal to the intercept, i.e., 1.T @ f(x) will not necessarily be 0. Hence, confidence intervals will usually
  be wider when using ConstType.DIFF (also when using ConstType.DROP, for the same reason) instead of ConstType.QR! Simulations reveal that for the same smoothing problem, the difference
  between ConstType.DIFF and ConstType.QR becomes smaller when increasing the number of basis functions. Intuitively, with many basis functions the dependency of the estimated f(x) on the
  intercept is usually reduced suffciently, so that the confidence intervals obtained with ConstType.DIFF match those achieved with ConstType.QR. In that case, the CIs achieved with ConstType.QR
  and ConstType.DIFF are usually substantially narrower than those obtained with ConstType.DROP.

  From this, it follows that:
    - ConstType.QR is preferred if computational efficiency is not crucial
    - ConstType.DROP is preferred if k is small (5-15) and computational efficiency is crucial
    - ConstType.DIFF is preferred if k is large (> 15) and computational efficiency is crucial

  A final note regards the use of tensor smooths when te==False. Since the value of any constant estimated for a smooth depends on the type of constraint used, the mmarginal functions estimated
  for the "main effects" (f(x),f(z)) and "interaction effect" (f(x,z)) in a model: y = a + f(x) + f(z) + f(x,z) will differ depending on the type of constraint used. The "Anova-like" decomposition
  described in detail in Wood (2017) is achievable only when using ConstType.QR.

  References:
  - Wood, S. N. (2020). Inference and computation with generalized additive models and their extensions. TEST, 29(2), 307–339. https://doi.org/10.1007/s11749-020-00711-5
  - Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. Statistical Science, 11(2), 89–121. https://doi.org/10.1214/ss/1038425655

  """
  Z: np.array or int = None
  type:ConstType = None