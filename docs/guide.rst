Programmer's Guide
==================

Extending the functionality of ``mssm`` is quite easy. Adding new families generally only requires implementing a specific base class and this is also the case for
penalty matrices. Implementing a new smooth basis can be achieved by adding a single function. Below we discuss these steps and provide code examples, outlining how
each of them can be achieved.

Adding a New ``Family``
----------------

Below we list the families already implemented, which can be estimated via the `GAMM` class:

- Gaussian (G)AMMs (:func:`mssm.src.python.exp_fam.Gaussian`)
- Gamma GAMMs (:func:`mssm.src.python.exp_fam.Gamma`)
- Inverse Gaussian GAMMs (:func:`mssm.src.python.exp_fam.InvGauss`)
- Binomial GAMMs (:func:`mssm.src.python.exp_fam.Binomial`)
- Poisson GAMMs (:func:`mssm.src.python.exp_fam.Poisson`)

Now, if you want to add a new ``Family``, your new family will have to implement the :func:`mssm.src.python.exp_fam.Family` base or template class. The documentation of the latter
provides details on all the methods your family will have to implement. You can also check the source code of the Families already implemented to get examples!

Adding a New ``GAMLSSFamily``
----------------

Below we list the GAMMLSS families already implemented, which can be estimated via the `GAMMLSS` class:

- Gaussian models of location and scale (:func:`mssm.src.python.exp_fam.GAUMLSS`)
- Gamma models of location and scale (:func:`mssm.src.python.exp_fam.GAMMALS`)
- Multinomial models (:func:`mssm.src.python.exp_fam.MULNOMLSS`)

Now, if you want to add a new ``GAMLSSFamily``, your new family will have to implement the :func:`mssm.src.python.exp_fam.GAMLSSFamily` base or template class. The documentation of the latter
provides details on all the methods your family will have to implement. You can also check the source code of the Families already implemented to get examples!

Adding a New ``GSMMFamily``
----------------

Below we list the GSMM families already implemented, which can be estimated via the `GSMM` class:

- Cox proportional Hazard models (:func:`mssm.src.python.exp_fam.PropHaz`)

To implement a new  member of the most general kind of smooth model, you will only need to implement the :func:`mssm.src.python.exp_fam.GSMMFamily` template class - ``mssm`` even supports completely derivative-free estimation.
You can check the :class:`mssm.models.GSMM` documentation for an example or the tutorial included with this documentation - it contains step-by-step instructions on how to implement this family.

Adding a New Marginal Smooth Basis
----------------

Adding a new marginal smooth basis only requires adding a single new function. This function interacts with ``mssm`` at three points: it is passed to the constructor of an instance of either the :class:`mssm.src.python.terms.f` class, the
:class:`mssm.src.python.terms.fs` class, or the :class:`mssm.src.python.terms.irf` class. Each of those takes a ``basis`` keyword argument, accepting a ``Callable`` - so a function. By default the ``basis`` argument
is set to ``basis=mssm.src.python.smooths.B_spline_basis`` - i.e., it receives the :func:`mssm.src.python.smooths.B_spline_basis` function as argument and calls it whenever the basis needs to be evaluated.

Speaking of evaluation. Every basis function needs to accept a couple of mandatory function arguments. Specifically, the function header needs to look something like this::

    def my_basis(cov:np.ndarray, event_onset:int|None, nk:int, min_c:float|None=None, max_c:float|None=None, **kwargs) -> np.ndarray:

Let's take a look at those mandatory arguments:

- ``cov``: This is set to the flattened covariate numpy array (i.e., of shape (-1,)) by ``mssm``.
- ``event_onset``: This is an integer or ``None``. If it's an integer, it reflects the sample on which to place a dirac delta with which the bases should be convolved - this is required if your basis is to work with the :class:`mssm.src.python.terms.irf` class. The :class:`mssm.src.python.terms.f` and :class:`mssm.src.python.terms.fs` classes always pass `None` to this argument.
- ``nk``: This is an integer corresponding to the number of basis functions to create.
- ``min_c``: This is the minimum covariate value, as a float, passed along by the :class:`mssm.src.python.terms.f` and :class:`mssm.src.python.terms.fs` classes. The :class:`mssm.src.python.terms.irf` class first checks if this argument is specified in ``basis_kwargs`` (more on that in a minute) and if so, passes along the value specified there.
- ``max_c``: Maximum covariate value, handled exactly like ``min_c``.

You can also set up your basis function to accept optional keyword arguments. For example, the function header of the default B-spline basis looks like this::

    B_spline_basis(cov:np.ndarray, event_onset:int|None, nk:int, min_c:float|None=None, max_c:float|None=None, drop_outer_k:bool=False, convolve:bool=False, deg:int=3) -> np.ndarray:

``deg`` here for example corresponds to the degree of the B-spline basis that should be created. How do these extra arguments get passed to the basis function?
The :class:`mssm.src.python.terms.f`, :class:`mssm.src.python.terms.fs`, and :class:`mssm.src.python.terms.irf` classes all accept a dictionary for the ``basis_kwargs`` argument which can be filled with values for the
optional extra arguments. For example, the code below creates a smooth function of variable "x1" (i.e., an instance of :class:`mssm.src.python.terms.f`) that relies on a B-spline basis of ``deg=3``::

    f(["x1"],basis=mssm.src.python.smooths.B_spline_basis,basis_kwargs={"deg":3})

Now, let's talk about the expected output from a basis function. The function header definition above informs us that a basis function should return a ``np.ndarray``. Specifically, it needs to be a two-dimensional
numpy array. The first dimension needs to be of the same length as ``cov`` (the covariate over which to evaluate the function) and the second dimension needs to be of length ``nk``. In other words, the output array
needs to hold in each column one of the ``nk`` basis functions, each evaluated over all values in ``cov``.

Adding a New Penalty Matrix
----------------

To create a new type of Penalty matrix, you need to implement the :class:`mssm.src.python.penalties.Penalty` class. Currently, mssm supports the following implementations: :class:`mssm.src.python.penalties.IdentityPenalty` and
:class:`mssm.src.python.penalties.DifferencePenalty`.  This penalty class interacts with ``mssm`` at three points: it is passed to the constructor of an instance of either the :class:`mssm.src.python.terms.f` class, the
:class:`mssm.src.python.terms.fs` class, or the :class:`mssm.src.python.terms.irf` class. Each of those takes a ``penalty`` keyword argument, accepting a list of instances of the :class:`mssm.src.python.penalties.Penalty` class (a list because
terms might have more than one penalty applied to them).

The implementation of the :class:`mssm.src.python.penalties.Penalty` class is quite simple, and looks like this::

    class Penalty:

        def __init__(self,pen_type:PenType) -> None:
            self.type = pen_type
        
        def constructor(self,n:int,constraint:ConstType|None,*args,**kwargs) -> tuple[list[float],list[int],list[int],list[float],list[int],list[int],int]:
            pass

The ``__init__`` method receives only a single argument, a :class:`mssm.src.python.custom_types.PenType` (see the documentation for supported values). For example, the ``pen_type`` of the  :class:`mssm.src.python.penalties.DifferencePenalty` is simply set
to ``PenType.DIFFERENCE``, while the ``__init__`` method of :class:`mssm.src.python.penalties.DifferencePenalty` accepts both ``PenType.IDENTITY`` and ``PenType.DISTANCE``. If you want to implement a derivative-based penalty, there is a ``PenType`` for that: ``PenType.DERIVATIVE``.

The actual construction of the penalty matrix is then handled by the ``constructor`` method. This is were you will want to implement the code that sets up your penalty matrix. The method receives two mandatory arguments:

- ``n``: An integer, corresponding to the dimension of the the square penalty matrix
- ``constraint``: Any constraint to absorb by the penalty or ``None`` if no constraint is required. If this argument is not ``None``, it will be an instance of the :class:`mssm.src.python.custom_types.Constraint` class, which holds all the information you need to absorb the constraint into the penalty (see the documentation).

Your ``constructor`` method can also accept additional arguments and key-word arguments. For example, the method header of the ``constructor`` method of the :class:`mssm.src.python.penalties.DifferencePenalty` class looks like this::

    constructor(self, n:int, constraint:ConstType|None, m:int=2) -> tuple[list[float],list[int],list[int],list[float],list[int],list[int],int]:

``m`` here corresponds to the order of the difference penalty. How do these extra arguments get passed to the basis function?
The :class:`mssm.src.python.terms.f`, :class:`mssm.src.python.terms.fs`, and :class:`mssm.src.python.terms.irf` classes all accept a list of dictionaries for the ``pen_kwargs`` argument - one for each penalty included in the list passed to the terms ``penalty`` argument.
Each dictionary can be filled with values for the optional extra arguments that should be passed to the ``constructor`` method of the corresponding instance of the :class:`mssm.src.python.penalties.Penalty` class.
For example, the code below creates a smooth function of variable "x1" (i.e., an instance of :class:`mssm.src.python.terms.f`) that relies on a B-spline basis of ``deg=3`` and is subjected to a single difference penalty (i.e., an instance of :class:`mssm.src.python.penalties.DifferencePenalty`)
of order ``m=2``::

    f(["x1"],basis=mssm.src.python.smooths.B_spline_basis,basis_kwargs={"deg":3},
      penalty=[DifferencePenalty()],pen_kwargs=[{"m":2}])

Now, let's talk about the expected output from the ``constructor`` method. Generally, this method needs to return the penalty matrix, a (matrix) root of the penalty matrix, and the rank of the penalty matrix. The rank is simply returned as an integer, but matters are slightly more
complicated for the two matrices: each of those needs to be returned in form of three lists. The first needs to hold non-zero values of the (root of the) penalty, the second needs to hold the row indices of these non-zero values, and the third needs to hold the column indices of these non-zero values. Hence,
the constructor needs to return six lists in total and an integer. The first three lists need to belong to the penalty matrix, while the last three lists need to belong to the penalty matrix root.