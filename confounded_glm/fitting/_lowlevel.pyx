"""
LBFGS and OWL-QN optimization algorithms

Python wrapper around liblbfgs.
"""

cimport numpy as np
import numpy as np
import warnings

np.import_array()   # initialize Numpy

ctypedef enum LineSearchAlgo :
    LBFGS_LINESEARCH_DEFAULT = 0,
    LBFGS_LINESEARCH_MORETHUENTE = 0,
    LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1,
    LBFGS_LINESEARCH_BACKTRACKING = 2,
    LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2,
    LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3,

ctypedef enum ReturnCode:
    LBFGS_SUCCESS = 0,
    LBFGS_CONVERGENCE = 0,
    LBFGS_STOP,
    LBFGS_ALREADY_MINIMIZED,
    LBFGSERR_UNKNOWNERROR = -1024,
    LBFGSERR_LOGICERROR,
    LBFGSERR_OUTOFMEMORY,
    LBFGSERR_CANCELED,
    LBFGSERR_INVALID_N,
    LBFGSERR_INVALID_N_SSE,
    LBFGSERR_INVALID_X_SSE,
    LBFGSERR_INVALID_EPSILON,
    LBFGSERR_INVALID_TESTPERIOD,
    LBFGSERR_INVALID_DELTA,
    LBFGSERR_INVALID_LINESEARCH,
    LBFGSERR_INVALID_MINSTEP,
    LBFGSERR_INVALID_MAXSTEP,
    LBFGSERR_INVALID_FTOL,
    LBFGSERR_INVALID_WOLFE,
    LBFGSERR_INVALID_GTOL,
    LBFGSERR_INVALID_XTOL,
    LBFGSERR_INVALID_MAXLINESEARCH,
    LBFGSERR_INVALID_ORTHANTWISE,
    LBFGSERR_INVALID_ORTHANTWISE_START,
    LBFGSERR_INVALID_ORTHANTWISE_END,
    LBFGSERR_OUTOFINTERVAL,
    LBFGSERR_INCORRECT_TMINMAX,
    LBFGSERR_ROUNDING_ERROR,
    LBFGSERR_MINIMUMSTEP,
    LBFGSERR_MAXIMUMSTEP,
    LBFGSERR_MAXIMUMLINESEARCH,
    LBFGSERR_MAXIMUMITERATION,
    LBFGSERR_WIDTHTOOSMALL,
    LBFGSERR_INVALIDPARAMETERS,
    LBFGSERR_INCREASEGRADIENT


cdef extern from "lbfgs.h":
    ctypedef double lbfgsfloatval_t
    ctypedef lbfgsfloatval_t* lbfgsconst_p "const lbfgsfloatval_t *"

    ctypedef lbfgsfloatval_t (*lbfgs_evaluate_t)(void *, lbfgsconst_p,
                              lbfgsfloatval_t *, int, lbfgsfloatval_t)
    ctypedef int (*lbfgs_progress_t)(void *, lbfgsconst_p, lbfgsconst_p,
                                     lbfgsfloatval_t, lbfgsfloatval_t,
                                     lbfgsfloatval_t, lbfgsfloatval_t,
                                     int, int, int)
 
    ctypedef struct lbfgs_parameter_t:
        int m
        lbfgsfloatval_t epsilon
        int past
        lbfgsfloatval_t delta
        int max_iterations
        LineSearchAlgo linesearch
        int max_linesearch
        lbfgsfloatval_t min_step
        lbfgsfloatval_t max_step
        lbfgsfloatval_t ftol
        lbfgsfloatval_t wolfe
        lbfgsfloatval_t gtol
        lbfgsfloatval_t xtol
        lbfgsfloatval_t orthantwise_c
        int orthantwise_start
        int orthantwise_end

    ReturnCode lbfgs(int, lbfgsfloatval_t *, lbfgsfloatval_t *, lbfgs_evaluate_t,
                     lbfgs_progress_t, void *, lbfgs_parameter_t *)

    void lbfgs_parameter_init(lbfgs_parameter_t *)
    lbfgsfloatval_t *lbfgs_malloc(int)
    void lbfgs_free(lbfgsfloatval_t *)


cdef class CallbackData(object):
    cdef object eval_fn
    cdef object progress_fn
    cdef object extra

    def __init__(self, eval_fn, progress_fn, extra):
        self.eval_fn = eval_fn
        self.progress_fn = progress_fn
        self.extra = extra


# Callback into Python evaluation callable.
cdef lbfgsfloatval_t call_eval(void *cb_data_v,
                               lbfgsconst_p x, lbfgsfloatval_t *g,
                               int n, lbfgsfloatval_t step):
    cdef object cb_data
    cdef np.npy_intp tshape[1]

    callback_data = <object>cb_data_v
    (f, progress_fn, shape, args) = callback_data
    tshape[0] = <np.npy_intp>n
    x_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE, <void *>x)
    g_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE, <void *>g)

    return f(x_array.reshape(shape), g_array.reshape(shape), *args)


# Callback into Python progress reporting callable.
cdef int call_progress(void *cb_data_v,
                       lbfgsconst_p x, lbfgsconst_p g,
                       lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls):
    cdef object cb_data
    cdef np.npy_intp tshape[1]

    callback_data = <object>cb_data_v
    (f, progress_fn, shape, args) = callback_data

    if progress_fn:
        tshape[0] = <np.npy_intp>n
        x_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE,
                                               <void *>x)
        g_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE,
                                               <void *>g)

        r = progress_fn(x_array.reshape(shape), g_array.reshape(shape), fx,
                        xnorm, gnorm, step, k, ls, *args)
        # TODO what happens when the callback returns the wrong type?
        return 0 if r is None else r
    else:
        return 0


# Copy an ndarray to a buffer allocated with lbfgs_malloc; needed to get the
# alignment right for SSE instructions.
cdef lbfgsfloatval_t *aligned_copy(x) except NULL:
    n = x.shape[0]
    x_copy = lbfgs_malloc(n)
    if x_copy is NULL:
        raise MemoryError
    for i in xrange(n):
        x_copy[i] = x[i]
    return x_copy


_LINE_SEARCH_ALGO = {
    'default' : LBFGS_LINESEARCH_DEFAULT,
    'morethuente' : LBFGS_LINESEARCH_MORETHUENTE,
    'armijo' : LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
    'wolfe' : LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
    'strongwolfe' : LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE
}


_ERROR_MESSAGES = {
    LBFGSERR_UNKNOWNERROR: "Unknown error." ,
    LBFGSERR_LOGICERROR: "Logic error.",
    LBFGSERR_OUTOFMEMORY: "Insufficient memory.",
    LBFGSERR_CANCELED: "The minimization process has been canceled.",
    LBFGSERR_INVALID_N: "Invalid number of variables specified.",
    LBFGSERR_INVALID_N_SSE: "Invalid number of variables (for SSE) specified.",
    LBFGSERR_INVALID_X_SSE: "The array x must be aligned to 16 (for SSE).",
    LBFGSERR_INVALID_EPSILON: "Invalid parameter epsilon specified.",
    LBFGSERR_INVALID_TESTPERIOD: "Invalid parameter past specified.",
    LBFGSERR_INVALID_DELTA: "Invalid parameter delta specified.",
    LBFGSERR_INVALID_LINESEARCH: "Invalid parameter linesearch specified.",
    LBFGSERR_INVALID_MINSTEP: "Invalid parameter max_step specified.",
    LBFGSERR_INVALID_MAXSTEP: "Invalid parameter max_step specified.",
    LBFGSERR_INVALID_FTOL: "Invalid parameter ftol specified.",
    LBFGSERR_INVALID_WOLFE: "Invalid parameter wolfe specified.",
    LBFGSERR_INVALID_GTOL: "Invalid parameter gtol specified.",
    LBFGSERR_INVALID_XTOL: "Invalid parameter xtol specified.",
    LBFGSERR_INVALID_MAXLINESEARCH:
        "Invalid parameter max_linesearch specified.",
    LBFGSERR_INVALID_ORTHANTWISE: "Invalid parameter orthantwise_c specified.",
    LBFGSERR_INVALID_ORTHANTWISE_START:
        "Invalid parameter orthantwise_start specified.",
    LBFGSERR_INVALID_ORTHANTWISE_END:
        "Invalid parameter orthantwise_end specified.",
    LBFGSERR_OUTOFINTERVAL:
        "The line-search step went out of the interval of uncertainty.",
    LBFGSERR_INCORRECT_TMINMAX:
        "A logic error occurred;"
        " alternatively, the interval of uncertainty became too small.",
    LBFGSERR_ROUNDING_ERROR:
        "A rounding error occurred;"
        " alternatively, no line-search step satisfies"
        " the sufficient decrease and curvature conditions.",
    LBFGSERR_MINIMUMSTEP: "The line-search step became smaller than min_step.",
    LBFGSERR_MAXIMUMSTEP: "The line-search step became larger than max_step.",
    LBFGSERR_MAXIMUMLINESEARCH:
        "The line-search routine reaches the maximum number of evaluations.",
    LBFGSERR_MAXIMUMITERATION:
        "The algorithm routine reaches the maximum number of iterations.",
    LBFGSERR_WIDTHTOOSMALL:
        "Relative width of the interval of uncertainty is at most xtol.",
    LBFGSERR_INVALIDPARAMETERS:
        "A logic error (negative line-search step) occurred.",
    LBFGSERR_INCREASEGRADIENT:
        "The current search direction increases the objective function value.",
}


class LBFGSError(Exception):
    pass


cdef class LBFGS(object):
    """LBFGS algorithm, wrapped in a class to permit setting parameters"""

    cdef lbfgs_parameter_t params

    def __init__(self):
        lbfgs_parameter_init(&self.params)

    LINE_SEARCH_ALGORITHMS = _LINE_SEARCH_ALGO.keys()

    property m:
        def __get__(self) :
            return self.params.m

        def __set__(self, int val):
            self.params.m = val

    property epsilon:
        def __get__(self) :
            return self.params.epsilon

        def __set__(self, double val):
            self.params.epsilon = val

    property past:
        def __get__(self) :
            return self.params.past

        def __set__(self, int val):
            self.params.past = val

    property delta:
        def __get__(self) :
            return self.params.delta

        def __set__(self, double val):
            self.params.delta = val

    property max_iterations:
        def __get__(self) :
            return self.params.max_iterations

        def __set__(self, int val):
            self.params.max_iterations = val

    property linesearch:
        def __get__(self) :
            return self.params.linesearch

        def __set__(self, algorithm):
            self.params.linesearch = _LINE_SEARCH_ALGO[algorithm]

    property max_linesearch:
        def __get__(self) :
            return self.params.max_linesearch

        def __set__(self, int val):
            self.params.max_linesearch = val

    property min_step:
        def __get__(self) :
            return self.params.min_step

        def __set__(self, double val):
            self.params.min_step = val

    property max_step:
        def __get__(self) :
            return self.params.max_step

        def __set__(self, double val):
            self.params.max_step = val

    property ftol:
        def __get__(self) :
            return self.params.ftol

        def __set__(self, double val):
            self.params.ftol = val

    property gtol:
        def __get__(self) :
            return self.params.gtol

        def __set__(self, double val):
            self.params.gtol = val

    property xtol:
        def __get__(self) :
            return self.params.xtol

        def __set__(self, double val):
            self.params.xtol = val

    property wolfe:
        def __get__(self) :
            return self.params.wolfe

        def __set__(self, double val):
            self.params.wolfe = val

    property orthantwise_c:
        def __get__(self) :
            return self.params.orthantwise_c

        def __set__(self, double val):
            self.params.orthantwise_c = val

    property orthantwise_start:
        def __get__(self) :
            return self.params.orthantwise_start

        def __set__(self, int val):
            self.params.orthantwise_start = val

    property orthantwise_end:
        def __get__(self) :
            return self.params.orthantwise_end

        def __set__(self, int val):
            self.params.orthantwise_end = val

    def minimize(self, f, x0, progress=None, args=()):
        """Minimize a function using LBFGS or OWL-QN

        Parameters
        ----------
        f : callable(x, g, *args)
            Computes function to minimize and its gradient.
            Called with the current position x (a numpy.ndarray), a gradient
            vector g (a numpy.ndarray) to be filled in and *args.
            Must return the value at x and set the gradient vector g.
        x0 : array-like
            Initial values. A copy of this array is made prior to optimization.
        progress : callable(x, g, fx, xnorm, gnorm, step, k, num_eval, *args),
                   optional
            If not None, called at each iteration after the call to f with the
            current values of x, g and f(x), the L2 norms of x and g, the line
            search step, the iteration number, the number of evaluations at
            this iteration and args (see below).
            If the return value from this callable is not 0 and not None,
            optimization is stopped and LBFGSError is raised.
        args : sequence
            Arbitrary list of arguments, passed on to f and progress as *args.
        """

        cdef np.npy_intp n
        cdef int n_i
        cdef int r
        cdef lbfgsfloatval_t *x_a
        cdef lbfgsfloatval_t* fx_final = NULL

        if not callable(f):
            raise TypeError("f must be callable, got %s" % type(f))
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable, got %s" % type(f))

        x0 = np.atleast_1d(x0)
        n = np.product(x0.shape)

        cdef np.npy_intp tshape[1]
        tshape[0] = <np.npy_intp>n

        n_i = n
        if n_i != n:
            raise LBFGSError("Array of %d elements too large to handle" % n)

        x_a = aligned_copy(x0.ravel())

        try:
            callback_data = (f, progress, x0.shape, args)
            r = lbfgs(n, x_a, fx_final, call_eval,
                      call_progress, <void *>callback_data, &self.params)

            if r == LBFGS_SUCCESS or r == LBFGS_ALREADY_MINIMIZED:

                x_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE,
                                                       <void *>x_a).copy()

                return x_array.reshape(x0.shape)
            elif r in (LBFGSERR_ROUNDING_ERROR, LBFGSERR_MAXIMUMLINESEARCH) :
                warnings.warn(_ERROR_MESSAGES[r])
                x_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE,
                                                       <void *>x_a).copy()

                return x_array.reshape(x0.shape)
            elif r == LBFGSERR_OUTOFMEMORY:
                raise MemoryError
            else:
                raise LBFGSError(_ERROR_MESSAGES[r])

        finally:
            lbfgs_free(x_a)


"""
LBFGS and OWL-QN optimization algorithms

Python wrapper around liblbfgs.
"""



def fmin_lbfgs(f, x0, progress=None, args=(), orthantwise_c=0,
               orthantwise_start=0, orthantwise_end=-1, m=6,
               epsilon=1e-5, past=0, delta=0., max_iterations=0,
               line_search="default", max_linesearch=20, min_step=1e-20,
               max_step=1e+20, ftol=1e-4, wolfe=0.9, gtol=0.9, xtol=1e-30):
    """Minimize a function using LBFGS or OWL-QN

     Parameters
    ----------
    f : callable(x, g, *args)
        Computes function to minimize and its gradient.
        Called with the current position x (a numpy.ndarray), a gradient
        vector g (a numpy.ndarray) to be filled in and *args.
        Must return the value at x and set the gradient vector g.

    x0 : array-like
        Initial values. A copy of this array is made prior to optimization.

    progress : callable(x, g, fx, xnorm, gnorm, step, k, num_eval, *args),
               optional
        If not None, called at each iteration after the call to f with the
        current values of x, g and f(x), the L2 norms of x and g, the line
        search step, the iteration number, the number of evaluations at
        this iteration and args (see below).
        If the return value from this callable is not 0 and not None,
        optimization is stopped and LBFGSError is raised.

    args : sequence
        Arbitrary list of arguments, passed on to f and progress as *args.

    orthantwise_c: float, optional (default=0)
        Coefficient for the L1 norm of variables.
        This parameter should be set to zero for standard minimization
        problems. Setting this parameter to a positive value activates
        Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
        minimizes the objective function F(x) combined with the L1 norm |x|
        of the variables, {F(x) + C |x|}. This parameter is the coefficient
        for the |x|, i.e., C. As the L1 norm |x| is not differentiable at
        zero, the library modifies function and gradient evaluations from
        a client program suitably; a client program thus have only to return
        the function value F(x) and gradients G(x) as usual. The default value
        is zero. 
        
        If orthantwise_c is set, then line_search cannot be the default
        and must be one of 'armijo', 'wolfe', or 'strongwolfe'.

    orthantwise_start: int, optional (default=0)
        Start index for computing L1 norm of the variables.
        This parameter is valid only for OWL-QN method
        (i.e., orthantwise_c != 0). This parameter b (0 <= b < N)
        specifies the index number from which the library computes the
        L1 norm of the variables x,
            |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
        In other words, variables x_1, ..., x_{b-1} are not used for
        computing the L1 norm. Setting b (0 < b < N), one can protect
        variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
        regression) from being regularized. The default value is zero.

    orthantwise_end: int, optional (default=-1)
        End index for computing L1 norm of the variables.
        This parameter is valid only for OWL-QN method
        (i.e., orthantwise_c != 0). This parameter e (0 < e <= N)
        specifies the index number at which the library stops computing the
        L1 norm of the variables x,

    m: int, optional, default=6
        The number of corrections to approximate the inverse hessian matrix.
        The L-BFGS routine stores the computation results of previous `m`
        iterations to approximate the inverse hessian matrix of the current
        iteration. This parameter controls the size of the limited memories
        (corrections). The default value is 6. Values less than 3 are
        not recommended. Large values will result in excessive computing time.

    epsilon: float, optional (default=1e-5)
        Epsilon for convergence test.
        This parameter determines the accuracy with which the solution is to
        be found. A minimization terminates when
            ||g|| < \ref epsilon * max(1, ||x||),
        where ||.|| denotes the Euclidean (L2) norm. The default value is
        1e-5.

    past: int, optional (default=0)
        Distance for delta-based convergence test.
        This parameter determines the distance, in iterations, to compute
        the rate of decrease of the objective function. If the value of this
        parameter is zero, the library does not perform the delta-based
        convergence test. The default value is 0.

    delta: float, optional (default=0.)
        Delta for convergence test.
        This parameter determines the minimum rate of decrease of the
        objective function. The library stops iterations when the
        following condition is met:
            (f' - f) / f < delta,
        where f' is the objective value of `past` iterations ago, and f is
        the objective value of the current iteration.
        The default value is 0.

    max_iterations: int, optional (default=0)
        The maximum number of iterations. Setting this parameter to zero continues an
        optimization process until a convergence or error. The default value
        is 0.

    line_search: str, optional (default="default")
        The line search algorithm.
        This parameter specifies a line search algorithm to be used by the
        L-BFGS routine. Possible values are:

        - 'default': same as 'morethuente'
        - 'morethuente': Method proposed by More and Thuente
        - 'armijo': backtracking with Armijo's conditions
        - 'wolfe': backtracking with Wolfe's conditions
        - 'strongwolfe': backtracking with strong Wolfe's conditions

    max_linesearch: int, optional (default=20)
        The maximum number of trials for the line search.
        This parameter controls the number of function and gradients evaluations
        per iteration for the line search routine. The default value is 20.

    min_step: float, optional (default=1e-20)
        The minimum step of the line search routine.
        The default value is 1e-20. This value need not be modified unless
        the exponents are too large for the machine being used, or unless the
        problem is extremely badly scaled (in which case the exponents should
        be increased).

    max_step: float, optional (default=1e20)
        The maximum step of the line search.
        The default value is 1e+20. This value need not be modified unless
        the exponents are too large for the machine being used, or unless the
        problem is extremely badly scaled (in which case the exponents should
        be increased).

    ftol: float, optional (default=1e-4)
        A parameter to control the accuracy of the line search routine.
        The default value is 1e-4. This parameter should be greater
        than zero and smaller than 0.5.

    wolfe: float, optional (default=0.9)
        A coefficient for the Wolfe condition. This parameter is valid only
        when the backtracking line-search algorithm is used with the Wolfe
        condition (`line_search='wolfe'` or `line_search='strongwolfe'`),
        The default value is 0.9. This parameter should be greater
        the `ftol` parameter and smaller than 1.0.

    gtol: float, optional (default=0.9)
        A parameter to control the accuracy of the line search routine.
        The default value is 0.9. If the function and gradient
        evaluations are inexpensive with respect to the cost of the
        iteration (which is sometimes the case when solving very large
        problems) it may be advantageous to set this parameter to a small
        value. A typical small value is 0.1. This parameter should be
        greater than the ftol parameter (1e-4) and smaller than
        1.0.


    xtol: float, optional (default=1e-30)
        The machine precision for floating-point values.
        This parameter must be a positive value set by a client program to
        estimate the machine precision. The line search routine will terminate
        with the status code (::LBFGSERR_ROUNDING_ERROR) if the relative width
        of the interval of uncertainty is less than this parameter.


    """

    # Some input validation to make sure defaults with OWL-QN are adapted correctly
    assert orthantwise_c >= 0, "Orthantwise_c cannot be negative"

    if orthantwise_c > 0 and line_search != 'wolfe':
        line_search = 'wolfe'
        warnings.warn("When using OWL-QN, 'wolfe' is the only valid line_search. line_search has been set to 'wolfe'.")

    opt = LBFGS()
    opt.orthantwise_c = orthantwise_c
    opt.orthantwise_start = orthantwise_start
    opt.orthantwise_end = orthantwise_end
    opt.m = m
    opt.epsilon = epsilon
    opt.past = past
    opt.delta = delta
    opt.max_iterations = max_iterations
    opt.linesearch = line_search
    opt.max_linesearch = max_linesearch
    opt.min_step = min_step
    opt.max_step = max_step
    opt.ftol = ftol
    opt.wolfe = wolfe
    opt.gtol = gtol
    opt.xtol = xtol

    return opt.minimize(f, x0, progress=progress, args=args)


