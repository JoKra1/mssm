from dataclasses import dataclass
from enum import Enum
import scipy as scp
import numpy as np
import warnings

try:
    # See if jax is installed to have AD support
    import jax
    import jax.numpy as jnp

    # Set jax to use 64 bit on import
    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


####################################### Enums/Types ####################################### # noqa
class PenType(Enum):
    """Custom Penalty data type used by internal functions."""

    IDENTITY = 1
    DIFFERENCE = 2
    DISTANCE = 3
    REPARAM1 = 4  # Approximate Demmler & Reinsch (1975) parameterization for univariate smooths;
    REPARAM2 = 10  # Natural parameterization by Wood (2006) for tensor smooths;
    NULL = 5
    DERIVATIVE = 6
    COEFFICIENTS = 7
    CUSTOM = 8
    SHARED = 9


class ConstType(Enum):
    """Custom Constraint data type used by internal functions."""

    DROP = 1
    QR = 2
    DIFF = 3


class VarType(Enum):
    """Custom variable data type used by internal functions."""

    NUMERIC = 1
    FACTOR = 2


class TermType(Enum):
    """Custom Term data type used by internal functions."""

    IRSMOOTH = 1
    SMOOTH = 2
    LINEAR = 3
    RANDINT = 4
    RANDSLOPE = 5


class DerivOrder(Enum):
    """Custom partial derivative order type. ``d1`` is first-order, ``d2`` is pure second
    derivatives, ``d2m`` is mixed second derivatives.
    """

    d1 = 1
    d2 = 2
    d2m = 3


####################################### Data-classes ####################################### # noqa


class ModeAD(Enum):
    """Custom data type indicating which AD mode to try. 1 is jit, 2 is eager, and 3 is finite
    differencing (not really AD).
    """

    jit = 1
    eager = 2
    fd = 3


class CacheAD:
    """Storage term for jax jitted likelihood, gradient, and hessian column functions or their
    eager counterparts if jit failed.

    ``GSMMFamily`` tries jit, but falls back to eager execution if that fails (e.g., due to
    python/numpy code that conflicts with jit, such as while loops). If both fail, `use_ad` is set
    to True and ``GSMMFamily`` falls back to finite difference approximations.

    see: https://docs.jax.dev/en/latest/201/control-flow.html for differences for when jit can fail.

    :param f: A function to evaluate the log-likelihood with signature ``f(pars, extra, ys)`` where
        ``ys`` are the reponse variables, ``pars`` are the vectors holding the parameters of the
        log-likelihood (i.e., the model matrix by coef vector products pushed through the inverse
        of the corresponding link function), and ``extra`` are any extra coefficients of the log
        likelihood.
    :type f: callable
    :param mode: Indicates which mode :func:`CacheAD.call` should rely on.
    :type mode: ModeAD
    :ivar list[callable] | None jitted: A list with three jitted functions for log-likelihood,
        gradient, and hessian vector product needed to get column j of hessian. Set to None in case
        jit fails or ``mode != ModeAd.jit``
    :ivar list[callable] | None eager: A list with three jax functions for log-likelihood,
        gradient, and hessian vector product needed to get column j of hessian via AD but for eager
        evaluation. Set to None in case eager execution fails or ``mode == ModeAd.fd``
    :ivar ModeAD mode: A :class:`ModeAD` instance indicating which mode :func:`CacheAD.call` should
        rely on. Set to ``mode`` but modified in case jit/eager fail.
    """

    def __init__(self, f: callable, mode: ModeAD = ModeAD.jit):

        self.jitted: list[callable] | None = None
        self.eager: list[callable] | None = None
        self.mode = mode

        # Catch errors that indicate something fundamentally opposing jax evalutation
        # see https://docs.jax.dev/en/latest/101/errors.html#errors
        self._bad_errors = (
            jax.errors.TracerArrayConversionError,
            jax.errors.TracerBoolConversionError,
            jax.errors.NonConcreteBooleanIndexError,
        )

        # Get gradient of llk with respect to pars and extra
        g = jax.grad(f, argnums=(0, 1))

        # Also need hessian vector product to get column of hessian
        def hvp(pars, extra, ys, v_pars, v_extra):
            # Forward-over-reverse: returns (gradient, Hessian @ [v_pars, v_extra]) where
            # the latter are basis vectors mapped to space of pars and extra
            return jax.jvp(lambda p, e: g(p, e, ys), (pars, extra), (v_pars, v_extra))

        if mode != ModeAD.fd:
            self.eager = [f, g, hvp]

            if mode == ModeAD.jit:
                self.jitted = [jax.jit(f), jax.jit(g), jax.jit(hvp)]

    def call(self, which: int, *args):
        """Tries calling ``self.jitted[which](*args)`` or ``self.eager[which](*args)``, depending
        on ``self.mode``.

        If calls fail and suggest something fundamentally opposed with JAX execution, ``self.mode``
        is automatically downranked from jitted -> eager -> fd to prevent further overhead of
        subsequent calls.

        :param which: Which function to call (0:llk, 1:grad, or 2:hessian vector product)
        :type which: int
        :return: The evaluated function or None in case both jit and eager execution failed.
        """

        if self.mode == ModeAD.jit:
            try:
                return self.jitted[which](*args)
            except self._bad_errors as e:
                # Catch errors suggesting something fundamentally opposing jit, in which case we
                # need to update mode
                warnings.warn(
                    f"Cannot jit function {which}. Fails with {e}. Falling back to eager execution."
                )
                self.mode = ModeAD.eager

        if self.mode == ModeAD.eager:
            try:
                return self.eager[which](*args)
            except self._bad_errors as e:
                # If bad errors persist, we have no choice but to fall back to fd.
                warnings.warn(
                    (
                        f"Eager execution of function {which} failed with {e}. "
                        "Falling back to finite differencing."
                    )
                )
                self.mode = ModeAD.fd

        return None


@dataclass
class LambdaTerm:
    """:math:`\\lambda` storage term.

    Usually ``model.overall_penalties`` holds a list of these.

    :ivar scp.sparse.csc_array S_J: The penalty matrix associated with this lambda term. Note, in
        case multiple penalty matrices share the same lambda value, the ``rep_sj`` argument
        determines how many diagonal blocks we need to fill with this penalty matrix to get
        ``S_J_emb``. Initialized with ``None``.
    :ivar scp.sparse.csc_array S_J_emb: A zero-embedded version of the penalty matrix associated
        with this lambda term. Note, this matrix contains ``rep_sj`` diagonal sub-blocks each
        filled with ``S_J``. Initialized with ``None``.
    :ivar scp.sparse.csc_array D_J_emb: Root of ``S_J_emb``, so that ``D_J_emb@D_J_emb.T=S_J_emb``.
        Note, that the root will not necessarily be square for merged penalties (that share a
        smoothing parameter via id). Initialized with ``None``.
    :ivar int rep_sj: How many sequential sub-blocks of ``S_J_emb`` need to be filled with ``S_J``.
        Useful if all levels of a categorical variable for which a separate smooth is to be
        estimated are assumed to share the same lambda value. Initialized with 1.
    :ivar float lam: The current estimate for :math:`\\lambda`. Initialized with 1.1.
    :ivar int start_index: The first row and column in the overall penalty matrix taken up by
        ``S_J``. Initialized with ``None``.
    :ivar PenType type: The type of this penalty term. Initialized with ``None``.
    :ivar int rank: The rank of ``S_J``. Initialized with ``None``.
    :ivar list[int] | int term: The index (indices) of the term(s) in a
        :class:`mssm.src.python.formula.Formula` with which this penalty is associated.
        Initialized with ``None``.
    :ivar bool frozen: Whether the penalty matrix with this term should be frozen during estimation
        (i.e., preventing any update to the penalty) or not. Initialized with ``False``, meaning
        the penalty is to be updated during estimation.
    """

    # Lambda term storage.
    # start_index can be useful in case we want to have multiple penalties on some
    # coefficients (see Wood, 2017; Wood & Fasiolo, 2017).
    S_J: scp.sparse.csc_array | None = None
    S_J_emb: scp.sparse.csc_array | None = None
    D_J_emb: scp.sparse.csc_array | None = None
    rep_sj: int = 1
    lam: float = 1.1
    start_index: int | None = None
    frozen: bool = False
    type: PenType | None = None
    rank: int | None = None
    term: list[int] | int | None = None
    clust_series: list[int] | None = None
    clust_weights: list[list[float]] | None = None
    dist_param: list[int] | int = 0
    rp_idx: int | None = None
    # Can hold multiple penalties associated with a single lambda value - need to keep track of
    # their individual matrices, start indices, etc. for easy splitting
    id: int | None = None
    S_Js: list[scp.sparse.csc_array] | None = None
    S_J_embs: list[scp.sparse.csc_array] | None = None
    D_J_embs: list[scp.sparse.csc_array] | None = None
    rep_sjs: list[int] | None = None
    start_indices: list[scp.sparse.csc_array] | None = None
    types: list[PenType] | None = None
    ranks: list[int] | None = None


@dataclass
class Reparameterization:
    """Holds information necessary to re-parameterize a smooth term.

    :ivar scp.sparse.csc_array Srp: The transformed penalty matrix
    :ivar scp.sparse.csc_array Drp: The root of the transformed penalty matrix
    :ivar scp.sparse.csc_array C: Transformation matrix for model matrix and/or penalty.
    """

    # Holds all information necessary to transform model matrix & penalty via various
    # re-parameterization strategies as discussed in Wood (2017).
    Srp: scp.sparse.csc_array | None = None  # Transformed penalty
    Drp: scp.sparse.csc_array | None = None  # Transformed root of penalty
    C: scp.sparse.csc_array | None = None
    scale: float | None = None
    IRrp: scp.sparse.csc_array | None = None
    rms1: float | None = None
    rms2: float | None = None
    rank: int | None = None


@dataclass
class Fit_info:
    """Holds information related to convergence (speed) for GAMMs, GAMMLSS, and GSMMs.

    :ivar int lambda_updates: The total number of lambda updates computed during estimation.
        Initialized with 0.
    :ivar int iter: The number of outer iterations (a single outer iteration can involve multiple
        lambda updates) completed during estimation. Initialized with 0.
    :ivar int code: Convergence status. Anything above 0 indicates that the model did not converge
        and estimates should be considered carefully. Initialized with 1.
    :ivar float eps: The fraction added to the last estimate of the negative Hessian of the
        penalized likelihood during GAMMLSS or GSMM estimation. If this is not 0 - the model should
        not be considered as converged, irrespective of what ``code`` indicates. This most likely
        implies that the model is not identifiable. Initialized with 0 and ignored for GAMM
        estimation.
    :ivar float K2: An estimate for the condition number of matrix ``A``, where ``A.T@A=H`` and
        ``H`` is the final estimate of the negative Hessian of the penalized likelihood.
        Only available if ``check_cond>0`` when ``model.fit()`` is called for any model
        (i.e., GAMM, GAMMLSS, GSMM). Initialized with ``None``.
    :ivar np.typing.NDArray[np.int_] dropped: The final set of coefficients dropped during
        GAMMLSS/GSMM estimation when using ``method in ["QR/Chol","LU/Chol","Direct/Chol"]`` or
        ``None`` in which case no coefficients were dropped. **Note**, if ``Q_emb`` is not ``None``
        these indices refer to the re-parameterized model. Initialized with None.
    :ivar scp.sparse.csc_array Q_emb: Optionally the transformation matrix of the
        re-parameterization used in the final GAMMLSS/GSMM iteration (i.e., ``Q_emb.T@S_emb@Q_emb``
        is the total penalty matrix after re-parameterization). Only set when coefficients were
        dropped in the final fitting iteration. Initialized with None.
    :ivar list[scp.sparse.csc_array] Qs: Optional list of transformation matrices ``Qd``
        corresponding to ``Q_emb`` (i.e., ``Xd@Qd`` is the model matrix for d-th linear predictor
        under repara). **Note**, only contains transformation matrices up to (and including) the
        last linear predictor that includes at least a single penalty. Initialized with None.
    """

    lambda_updates: int = 0
    iter: int = 0
    code: int = 1
    eps: float = 0
    K2: float | None = None
    dropped: np.typing.NDArray[np.int_] | None = None
    Q_emb: scp.sparse.csc_array | None = None
    Qs: list[scp.sparse.csc_array] | None = None


@dataclass
class Constraint:
    """
    Constraint storage. ``Z``, either holds the Qr-based correction matrix that needs to be
    multiplied with :math:`\\mathbf{X}`, :math:`\\mathbf{S}`, and :math:`\\mathbf{D}`
    (where :math:`\\mathbf{D}\\mathbf{D}^T = \\mathbf{S}`) to make terms subject to the
    conventional sum-to-zero constraints applied also in mgcv (Wood, 2017), the column/row that
    should be dropped from those - then :math:`\\mathbf{X}` can also no longer take
    on a constant, or ``None`` indicating that the model should be "difference re-coded" to enable
    sparse sum-to-zero constraints. The latter two are available in mgcv's ``smoothCon`` function
    by setting the ``sparse.cons`` argument to 1 or 2 respectively.

    The QR-based approach is described in detail by Wood (2017) and is similar to just mean
    centering every basis function involved in the smooth and then dropping one column from the
    corresponding centered model matrix. The column-dropping approach is self-explanatory. The
    difference re-coding re-codes bases functions to correspond to differences of bases functions.
    The resulting basis remains sparser than the alternatives, but this is not a true centering
    constraint: :math:`f(x)` will not necessarily be orthogonal to the intercept,
    i.e., :math:`\\mathbf{1}^T \\mathbf{f(x)}` will not necessarily be 0. Hence, confidence
    intervals will usually be wider when using ConstType.DIFF (also when using ConstType.DROP, for
    the same reason) instead of ``ConstType.QR`` (see Wood; 2017,2020)!

    A final note regards the use of tensor smooths when ``te==False``. Since the value of any
    constant estimated for a smooth depends on the type of constraint used, the marginal functions
    estimated for the "main effects" (:math:`f(x)`, :math:`f(z)`) and "interaction effect"
    (:math:`f(x,z)`) in a model: :math:`y = a + f(x) + f(z) + f(x,z)` will differ depending on the
    type of constraint used. The "Anova-like" decomposition described in detail in Wood (2017) is
    achievable only when using ``ConstType.QR``.

    Thus, ``ConstType.QR`` is the default by all ``mssm`` functions, and the other two options
    should be considered experimental.

    References:
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
        Edition (2nd ed.).
      - Wood, S. N. (2020). Inference and computation with generalized additive models and their \
        extensions. TEST, 29(2), 307–339. https://doi.org/10.1007/s11749-020-00711-5
      - Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. \
        Statistical Science, 11(2), 89–121. https://doi.org/10.1214/ss/1038425655

    """

    Z: np.ndarray | int | None = None
    type: ConstType | None = None


@dataclass
class SamplerResult:
    """Holds quantities sampled from the posterior of GAMM, GAMMLSS, or GSMM models.

    :ivar np.ndarray | None llks: An array holding the traces of the log-joint densities (up to a
        constant) of coefficients, optional scale or theta parameters, and optional
        :math:`\\rho = log(\\lambda)` log smoothing penalties for all chains sampled by a mcmc
        sampler. Dimension is ``(n_chains,n_samples,1)``. Initialized with None.
    :ivar np.ndarray | None coefs: An array holding the traces of the coefficients (i.e.,
        :math`\\boldsymbol{\\beta}`) for all chains sampled by a mcmc sampler. Dimension is
        ``(n_chains,n_samples,n_coef)``. Initialized with None.
    :ivar np.ndarray | None lscales: If the model sampled is a :class:`mssm.models.GAMM` with
        a scale parameter (i.e., :math:`\\phi`), then this array holds the traces of the **log**
        scale parameter for all chains sampled by a mcmc sampler. Dimension is
        ``(n_chains,n_samples,1)``. For all other models, this field is None. Initialized with None.
    :ivar np.ndarray | None thetas: If the model sampled is a :class:`mssm.models.GAMM` with
        family inheriting from :class:`mssm.src.python.exp_fam.ExtendedFamily`, then this array
        holds the traces of the theta parameters of the extended family for all chains sampled by a
        mcmc sampler. Dimension is ``(n_chains,n_samples,n_theta)``. For all other models, this
        field is None. Initialized with None.
    :ivar np.ndarray | None rhos: An array holding the traces of the log smoothing penalties (i.e.,
        :math`\\boldsymbol{\\rho}`) for all chains sampled by a mcmc sampler. Dimension is
        ``(n_chains,n_samples,n_rho)``. Note, that sampling these parameters is optional. Instead
        they are often fixed at their MAP. If they are not sampled, this field will simply be None.
        Initialized with None.
    """

    lps: np.ndarray | None = None
    coefs: np.ndarray | None = None
    lscales: np.ndarray | None = None
    thetas: np.ndarray | None = None
    rhos: np.ndarray | None = None
