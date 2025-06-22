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
  to be multiplied with :math:`\mathbf{X}`, :math:`\mathbf{S}`, and :math:`\mathbf{D}` (where :math:`\mathbf{D}\mathbf{D}^T = \mathbf{S}`) to make terms subject to the conventional sum-to-zero constraints
  applied also in mgcv (Wood, 2017), the column/row that should be dropped from those - then :math:`\mathbf{X}` can also no longer take
  on a constant, or ``None`` indicating that the model should be "difference re-coded" to enable sparse sum-to-zero
  constraints. The latter two are available in mgcv's ``smoothCon`` function by setting the ``sparse.cons`` argument to 1 or
  2 respectively.

  The QR-based approach is described in detail by Wood (2017) and is similar to just mean centering every basis function involved in the
  smooth and then dropping one column from the corresponding centered model matrix. The column-dropping approach is self-explanatory. The
  difference re-coding deserves some explanation. Consider the model:

  .. math::

    y = f(x) = b_1*f_1(x) + b_2*f_2(x) + b_3*f_3(x) + b_4*f_4(x) + b_5*f_5(x)
  
  In ``mssm`` :math:`f(x)` will be parameterized via five B-spline basis functions :math:`f_i(x)`, each weighted by a coefficient :math:`b_i`. Each of the :math:`f_i(x)` has
  support only over a small range of :math:`x` - resulting in a sparse model matrix :math:`\mathbf{X}`. This desirable property is lost when enforcing the conventional
  sum-to-zero constraints (either QR-based or by subtracting the mean of each basis over :math:`x` from the corresponding :math:`f_i(x)`). To maintain sparsity,
  one can enforce a coefficients-sum-to-zero constraint (Wood, 2020) by re-coding instead:

  Set :math:`b_1 = -b_2`

  Then, re-write:
  
  .. math::

    y = -b_2 * f_1(x) + b_2 * f_2(x) + b_3 * f_3(x) - b_3 * f_2(x) + b_4 * f_4(x) - b_4 * f_3(x) + b_5 * f_5(x) - b_5 * f_4(x)

    y = -b_2 * f_1(x) + (b_2 - b_3) * f_2(x) + (b_3 - b_4) * f_3(x) + (b_4 - b_5) * f_4(x) + b_5 * f_5(x)

    y = b_2 * (f_2(x) - f_1(x)) + b_3 * (f_3(x) - f_2(x)) + b_4 * (f_4(x) - f_3(x)) + b_5 * (f_5(x) - f_4(x))

  Line 3 shows how the constraint can be absorbed for model fitting by first computing new bases :math:`(f_i(x) - f_{i-1}(x))` and then estimating
  the coefficients based on those (this is implemented in mgcv's ``smoothCon`` function when setting ``sparse.cons=2``). Note that the constrained
  model, just like when using the QR-based or dropping approach, requires dropping one of the :math:`k` original coefficients for estimation since only :math:`k-1`
  coefficients are allowed to vary. The same sum-to-zero constraint can be achieved by fixing one of the central bases in the original model to it's neighbor (e.g., setting :math:`b_2 = -b_3`) or
  by setting :math:`b_1= -b_2 - b_3 - b_4 - b_5`. ``mssm`` fixes one of the central bases to it's neighbor.

  With a B-splines basis, it would be necessary that :math:`b_1 = b_2 = b_3 = b_4 = b_5` for the model to fit a constant :math:`f(x)` over all values of :math:`x` (Eilers & Marx, 1996).
  In the constrained model this is no longer possible due to :math:`b_1 = -b_2`, effectively removing the constant from the space of functions that :math:`f(x)` can approximate.
  Instead, in the constrained model if :math:`b_2 = b_3 = b_4 = b_5`, :math:`f(x)` summed over all :math:`x` will be equal to zero!
  
  Importantly, the "new" bases obtained by computing :math:`(f_i(x) - f_{i-1}(x))` are all still relatively sparse. This is because neighboring
  :math:`f_i(x)` overlap in their support and because the support of each individual :math:`f_i(x)` is narrow, i.e., :math:`f_i(x)` is sparse itself.

  However, this is not a true centering constraint: :math:`f(x)` will not necessarily be orthogonal to the intercept, i.e., :math:`\mathbf{1}^T \mathbf{f(x)}` will not necessarily be 0. Hence, confidence intervals will usually
  be wider when using ConstType.DIFF (also when using ConstType.DROP, for the same reason) instead of ``ConstType.QR`` (see Wood; 2017,2020)!

  A final note regards the use of tensor smooths when ``te==False``. Since the value of any constant estimated for a smooth depends on the type of constraint used, the mmarginal functions estimated
  for the "main effects" (:math:`f(x)`, :math:`f(z)`) and "interaction effect" (:math:`f(x,z)`) in a model: :math:`y = a + f(x) + f(z) + f(x,z)` will differ depending on the type of constraint used. The "Anova-like" decomposition
  described in detail in Wood (2017) is achievable only when using ``ConstType.QR``.

  References:
    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
    - Wood, S. N. (2020). Inference and computation with generalized additive models and their extensions. TEST, 29(2), 307–339. https://doi.org/10.1007/s11749-020-00711-5
    - Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. Statistical Science, 11(2), 89–121. https://doi.org/10.1214/ss/1038425655

  """
  Z: np.array or int = None
  type:ConstType = None