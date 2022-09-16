"""
A slight modification to Scipy's implementation of differential evolution. To speed up predictions, the entire parameters array is passed to `self.func`, where a neural network model can batch its computations and execute in parallel. Search for `CHANGES` to find all code changes.

Dan Kondratyuk 2018

Original code adapted from
https://github.com/scipy/scipy/blob/70e61dee181de23fdd8d893eaa9491100e2218d7/scipy/optimize/_differentialevolution.py
----------

differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import copy
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize.optimize import _status_message
from scipy._lib._util import check_random_state
#from scipy._lib.six import xrange, string_types
from six import string_types
import warnings

from scipy.optimize import dual_annealing # for local search

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps


def differential_evolution(func, bounds, args=(), strategy='best1bin',
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube', atol=0, iter_LS=7, DE='DE', LS=0, no_LS=True,
                           memory_size=100, p_best_rate=0.11, arc_rate=1):
    """Finds the global minimum of a multivariate function.
    Differential Evolution is stochastic in nature (does not use gradient
    methods) to find the minimium, and can search large areas of candidate
    space, but often requires larger numbers of function evaluations than
    conventional gradient based techniques.
    The algorithm is due to Storn and Price [1]_.
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.RandomState` singleton is used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with seed.
        If `seed` is already a `np.random.RandomState instance`, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(M, len(x))``, where len(x) is the number of parameters.
              `init` is clipped to `bounds` before use.
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random'
        initializes the population randomly - this has the drawback that
        clustering can occur, preventing the whole of parameter space being
        covered. Use of an array to specify a population subset could be used,
        for example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.  If `polish`
        was employed, and a lower minimum was obtained by the polishing, then
        OptimizeResult also contains the ``jac`` attribute.
    Notes
    -----
    Differential evolution is a stochastic population based method that is
    useful for global optimization problems. At each pass through the population
    the algorithm mutates each candidate solution by mixing with other candidate
    solutions to create a trial candidate. There are several strategies [2]_ for
    creating trial candidates, which suit some problems more than others. The
    'best1bin' strategy is a good starting point for many systems. In this
    strategy two members of the population are randomly chosen. Their difference
    is used to mutate the best member (the `best` in `best1bin`), :math:`b_0`,
    so far:
    .. math::
        b' = b_0 + mutation * (population[rand0] - population[rand1])
    A trial vector is then constructed. Starting with a randomly chosen 'i'th
    parameter the trial is sequentially filled (in modulo) with parameters from
    `b'` or the original candidate. The choice of whether to use `b'` or the
    original candidate is made with a binomial distribution (the 'bin' in
    'best1bin') - a random number in [0, 1) is generated.  If this number is
    less than the `recombination` constant then the parameter is loaded from
    `b'`, otherwise it is loaded from the original candidate.  The final
    parameter is always loaded from `b'`.  Once the trial candidate is built
    its fitness is assessed. If the trial is better than the original candidate
    then it takes its place. If it is also better than the best overall
    candidate it also replaces that.
    To improve your chances of finding a global minimum use higher `popsize`
    values, with higher `mutation` and (dithering), but lower `recombination`
    values. This has the effect of widening the search radius, but slowing
    convergence.
    .. versionadded:: 0.15.0
    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`.
    >>> from scipy.optimize import rosen, differential_evolution
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = differential_evolution(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)
    Next find the minimum of the Ackley function
    (http://en.wikipedia.org/wiki/Test_functions_for_optimization).
    >>> from scipy.optimize import differential_evolution
    >>> import numpy as np
    >>> def ackley(x):
    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    >>> bounds = [(-5, 5), (-5, 5)]
    >>> result = differential_evolution(ackley, bounds)
    >>> result.x, result.fun
    (array([ 0.,  0.]), 4.4408920985006262e-16)
    References
    ----------
    .. [1] Storn, R and Price, K, Differential Evolution - a Simple and
           Efficient Heuristic for Global Optimization over Continuous Spaces,
           Journal of Global Optimization, 1997, 11, 341 - 359.
    .. [2] http://www1.icsi.berkeley.edu/~storn/code.html
    .. [3] http://en.wikipedia.org/wiki/Differential_evolution
    """
    if DE == 'DE':
       solver = DifferentialEvolutionSolver(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,  
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback, 
                                         disp=disp, init=init, atol=atol, iter_LS=iter_LS, LS=LS, no_LS=no_LS)#
    elif DE == 'SHADE':
       solver = SHADE(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback,
                                         disp=disp, init=init, atol=atol, iter_LS=iter_LS, LS=LS, no_LS=no_LS,
                                         memory_size=memory_size, p_best_rate=p_best_rate, arc_rate=arc_rate)
    else:
       solver = EBLSHADE(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback,
                                         disp=disp, init=init, atol=atol, iter_LS=iter_LS, no_LS=no_LS, LS=LS,
                                         memory_size=memory_size, p_best_rate=p_best_rate, arc_rate=arc_rate)

    return solver.solve()


class DifferentialEvolutionSolver(object):

    """This class implements the differential evolution solver
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.random.RandomState` singleton is
        used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with `seed`.
        If `seed` is already a `np.random.RandomState` instance, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True, then `scipy.optimize.minimize` with the `L-BFGS-B` method
        is used to polish the best population member at the end. This requires
        a few more function evaluations.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(M, len(x))``, where len(x) is the number of parameters.
              `init` is clipped to `bounds` before use.
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random'
        initializes the population randomly - this has the drawback that
        clustering can occur, preventing the whole of parameter space being
        covered. Use of an array to specify a population could be used, for
        example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    """

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'currenttobest1bin': '_currenttobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    __init_error_msg = ("The population initialization method must be one of "
                        "'latinhypercube' or 'random', or an array of shape "
                        "(M, N) where N is the number of parameters and M>5")

    def __init__(self, func, bounds, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0, iter_LS=5, no_LS=True, LS=0):

        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.callback = callback
        self.polish = polish

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or not
                np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')
        self._bounds = bounds
        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = check_random_state(seed)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        # the minimum is 5 because 'best2bin' requires a population that's at
        # least 5 long
        self.num_population_members = max(5, popsize * self.parameter_count)

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        self._nfev = 0
        if isinstance(init, string_types):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'random':
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)

        self.disp = disp

        self.LS = LS
        self.iter_LS = 0
        self.no_LS = no_LS
        self.max_iter_LS = iter_LS
        self.last_LS = True # the last optimum is obtained by local search
        self.pos_LS = 0 # the position of population for local search 
        self.pop_size = popsize * len(bounds)
        self.DA_result = list()

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.random_sample(self.population_shape)

        # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_array(self, init):
        """
        Initialises the population with a user specified population.
        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (M, len(x)), where len(x) is the number of parameters.
            The population is clipped to the lower and upper `bounds`.
        """
        # make sure you're using a float array
        popn = np.asfarray(init)

        if (np.size(popn, 0) < 5 or
                popn.shape[1] != self.parameter_count or
                len(popn.shape) != 2):
            raise ValueError("The population supplied needs to have shape"
                             " (M, len(x)), where M > 4.")

        # scale values and clip to bounds, assigning to population
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)

        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        """
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        return (np.std(self.population_energies) /
                np.abs(np.mean(self.population_energies) + _MACHEPS))

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nit, warning_flag = 0, False
        status_message = _status_message['success']

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        current_best_prob = 1.0
        # do the optimisation.
        for nit in range(1, self.maxiter + 1):
            # evolve the population by a generation
            try:
                next(self)
                # checking if doing local search
                if self.LS >= 2 and current_best_prob > self.population_energies[0]: 
                   self._local_search(epoch=nit)
                   current_best_prob =  self.population_energies[0]
            except StopIteration:
                warning_flag = True
                status_message = _status_message['maxfev']
                break

            if self.disp:
                print("differential_evolution step %d: f(x)= %g"
                      % (nit,
                         self.population_energies[0]))

            # should the solver terminate?
            convergence = self.convergence

            if (self.callback and convergence != 0 and
                    self.callback(self._scale_parameters(self.population[0]),
                                  convergence=self.tol / convergence) is True):

                warning_flag = True
                status_message = ('callback function requested stop early '
                                  'by returning True')
                break

            intol = (np.std(self.population_energies) <=
                     self.atol +
                     self.tol * np.abs(np.mean(self.population_energies)))
            if warning_flag or intol:
                break

        else:
            status_message = _status_message['maxiter']
            warning_flag = True

        
        if  self.LS == 1:
           res = self._local_search(epoch=nit)
           #if len(self.DA_result) > 0:
           #    _sda = sorted(zip(np.array(self.DA_result)[:,0], np.array(self.DA_result)[:,1]), key=lambda x: x[1])[0]
           print('The best local search, prob=%f, with solution:'%res.fun, self._unscale_parameters(res.x))
        if self.LS >= 2:
           print("The local search best solutions are:")
           _sda = sorted(zip(np.array(self.DA_result)[:,0], np.array(self.DA_result)[:,1], np.array(self.DA_result)[:,2]), key=lambda x: x[1])
           for d in _sda:
             print(d[0], d[1][0], d[2])

        if self.convergence !=0 and len(self._bounds)==5:
            self._minimize_pixels()

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True))

        if self.polish:
            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method='L-BFGS-B',
                              bounds=self.limits.T,
                              args=self.args)

            self._nfev += result.nfev
            DE_result.nfev = self._nfev

            if result.fun < DE_result.fun:
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)
        return DE_result

    # compute the number of positions in the self.population
    def _num_pixels(self):
        parameters = np.array([self._scale_parameters(trial) for trial in self.population])[:,0:2].astype(int)
        return np.size(np.unique(parameters,axis=0),0)

    # In the case the DE find no solution, it finds a minimal number of pixels that are sufficient to fool the model    
    # It must be one-pixel attacking.
    def _minimize_pixels(self, ep=0.5, try_all=True):
        if try_all:
           parameters = np.reshape(np.array([self._scale_parameters(trial) for trial in self.population]),(1,len(self.population)*5))
           best_p = self.func(parameters, *self.args)
           print('The minimal number of successful attacking pixels is %d with probability %f'%(self._num_pixels(), best_p))
           return self._num_pixels()
        m=1
        best_p = self.population_energies[0]
        lenp = len(self.population_energies)
        sorted_index = np.array(sorted(zip(self.population_energies, np.arange(lenp)), key=lambda x: x[0]))[:,1].astype(int)
        si = np.zeros((lenp))
        for i in range(lenp):
            si[sorted_index[i]] = i
        trials = [self.population[0]]
        while m < lenp:
            trials.insert(0,self.population[m])
            #parameters = np.array([self._scale_parameters(trial) for trial in trials])
            parameters = np.reshape(np.array([self._scale_parameters(trial) for trial in trials]),(1,(m+1)*5))
            best_p = self.func(parameters, *self.args)
            if best_p <= ep: break
            m += 1
        print('The minimal number of successful attacking pixels is %d with probability %f'%(self._num_pixels(), best_p))
        return m

    def _func_DA(self, xs):
        return self.func(self._scale_parameters(xs))

    def _callback_DA(self, x, f, context=0):
        return self.callback(self._scale_parameters(x), convergence=self.tol / self.convergence )

    def _local_search(self, verbose=False, epoch=0, maxfun=1000, maxiter=500, b0=False):
        pos = 0
        x0=None
        if b0: x0=self.population[pos]
        res = dual_annealing(self._func_DA, bounds=[(.0, 1.)]*len(self._bounds), callback=self._callback_DA, x0=x0, maxiter=maxiter, maxfun=maxfun)
        if verbose:
            print('best_x:', res.x, 'best_y:', res.fun)
        if self.LS >= 2:
            self.DA_result.append([self._unscale_parameters(res.x), res.fun, epoch])
        if self.LS == 4 and res.fun < self.population_energies[pos]:
            self.population_energies[pos] =  res.fun[0]
            self.population[pos] = self._unscale_parameters(res.x)
        
        return res

    def _calculate_population_energies(self):
        """
        Calculate the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """

        ##############
        ## CHANGES: self.func operates on the entire parameters array
        ##############
        itersize = max(0, min(len(self.population), self.maxfun - self._nfev + 1))
        candidates = self.population[:itersize]
        parameters = np.array([self._scale_parameters(c) for c in candidates]) # TODO: can be vectorized
        energies = self.func(parameters, *self.args)
        self.population_energies = energies
        self._nfev += itersize

        # for index, candidate in enumerate(self.population):
        #     if self._nfev > self.maxfun:
        #         break

        #     parameters = self._scale_parameters(candidate)
        #     self.population_energies[index] = self.func(parameters,
        #                                                 *self.args)
        #     self._nfev += 1

        ##############
        ##############

        

        minval = np.argmin(self.population_energies)

        # put the lowest energy into the best solution position.
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy

        self.population[[0, minval], :] = self.population[[minval, 0], :]

    def __iter__(self):
        return self

    def __next__(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        if self.dither is not None:
            self.scale = (self.random_number_generator.rand()
                          * (self.dither[1] - self.dither[0]) + self.dither[0])

        ##############
        ## CHANGES: self.func operates on the entire parameters array
        ##############

        itersize = max(0, min(self.num_population_members, self.maxfun - self._nfev + 1))
        trials = np.array([self._mutate(c) for c in range(itersize)]) # TODO: can be vectorized
        for trial in trials: self._ensure_constraint(trial)
        parameters = np.array([self._scale_parameters(trial) for trial in trials])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize
        '''
        if min(energies) < self.population_energies[0]:
            do_local_search = True
        else:
            do_local_search = False
            self.iter_LS = 0
        self.iter_LS += 1
        '''
        for candidate,(energy,trial) in enumerate(zip(energies, trials)):
            # if the energy of the trial candidate is lower than the
            # original population member then replace it
            if energy < self.population_energies[candidate]:
                self.population[candidate] = trial
                self.population_energies[candidate] = energy

                # if the trial candidate also has a lower energy than the
                # best solution then replace that as well
                if energy < self.population_energies[0]:
                    self.population_energies[0] = energy
                    self.population[0] = trial
                    self.last_LS = False
        ''' 
        if do_local_search:
            self._local_search() 
        if self.iter_LS == self.max_iter_LS:  
            self._local_search() 
            self.last_LS = True
            self.iter_LS = 0
        '''
        # for candidate in range(self.num_population_members):
        #     if self._nfev > self.maxfun:
        #         raise StopIteration

        #     # create a trial solution
        #     trial = self._mutate(candidate)

        #     # ensuring that it's in the range [0, 1)
        #     self._ensure_constraint(trial)

        #     # scale from [0, 1) to the actual parameter value
        #     parameters = self._scale_parameters(trial)

        #     # determine the energy of the objective function
        #     energy = self.func(parameters, *self.args)
        #     self._nfev += 1

        #     # if the energy of the trial candidate is lower than the
        #     # original population member then replace it
        #     if energy < self.population_energies[candidate]:
        #         self.population[candidate] = trial
        #         self.population_energies[candidate] = energy

        #         # if the trial candidate also has a lower energy than the
        #         # best solution then replace that as well
        #         if energy < self.population_energies[0]:
        #             self.population_energies[0] = energy
        #             self.population[0] = trial

        ##############
        ##############

        return self.x, self.population_energies[0]

    def next(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # next() is required for compatibility with Python2.7.
        return self.__next__()

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        
        for index in np.where((trial < 0) | (trial > 1))[0]:
            trial[index] = self.random_number_generator.rand()            
        vectorized
        revised by Yisong
        """
        indx = (trial < 0) | (trial > 1)
        if sum(indx) > 0:
            trial[indx] = self.random_number_generator.random_sample(trial.shape)[indx]
        #return trial

    def _mutate(self, candidate):
        """
        create a trial vector based on a mutation strategy
        """
        trial = np.copy(self.population[candidate])

        rng = self.random_number_generator

        fill_point = rng.randint(0, self.parameter_count)

        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = rng.rand(self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            while (i < self.parameter_count and
                   rng.rand() < self.cross_over_probability):

                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        """
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        """
        randtobest1bin, randtobest1exp
        """
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] -
                                self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """
        currenttobest1bin, currenttobest1exp
        """
        r0, r1 = samples[:2]
        bprime = (self.population[candidate] + self.scale * 
                  (self.population[0] - self.population[candidate] +
                   self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs


class SHADE(DifferentialEvolutionSolver):
    DifferentialEvolutionSolver._binomial['currenttopbest1'] = '_currenttopbest1'
    '''
    One big mistake I made is '__init__' was written as '__init___'. 
    It reported that 'memory_size' is not a member of ELSHADE object.
    -YS, 2020.09.16
    '''
    def __init__(self, func, bounds, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0, iter_LS=5, no_LS=True,
                 memory_size=100, p_best_rate=0.11, arc_rate=1, LS=0):
        super().__init__(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback, LS=LS,
                                         disp=disp, init=init, atol=atol)
        
        self.memory_size = memory_size
        self.memory_f = np.ones((memory_size))*0.5
        self.memory_cr = np.ones((memory_size))*0.5
        self.memory_pos = 0

        #self.pop_size = popsize * self.parameter_count

        self.p_best_rate = p_best_rate # this is for JADE, which is static
        self.arc_rate = arc_rate

        self.archive_NP = arc_rate * self.pop_size 
        self.archive_pop = np.zeros((0,self.parameter_count))
        self.archive_funvalues = np.zeros((0))

    def __next__(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()
        """
        if self.dither is not None:
            self.scale = (self.random_number_generator.rand()
                          * (self.dither[1] - self.dither[0]) + self.dither[0]) 
        """
        itersize = max(0, min(self.num_population_members, self.maxfun - self._nfev + 1))

        #### mutation and crossover
        mem_rand_index = (self.memory_size * self.random_number_generator.rand(self.pop_size)).astype(int)

        mu_sf = self.memory_f[mem_rand_index]
        mu_cr = self.memory_cr[mem_rand_index]

        cr = self.random_number_generator.normal(mu_cr, 0.1)
        cr[ cr < 0 ] = 0.
        cr[ cr > 1 ] = 1.

        sf = mu_sf + 0.1 * np.tan(np.pi * (self.random_number_generator.rand(self.pop_size) - 0.5))
        sf[ sf > 1 ] = 1.
        pos = np.where(sf <= 0)[0]
        while np.size(pos) > 0:
            sf[pos] = mu_sf[pos] + 0.1 * np.tan(np.pi * (self.random_number_generator.rand(np.size(pos)) - 0.5))
            pos = np.where(sf <= 0)[0]

        ## computing the pbest vectors
        p_best_rate = self.random_number_generator.uniform(2./self.pop_size, 0.2, self.pop_size)
        nTop = np.array([max(2, x) for x in (p_best_rate * self.pop_size).round().astype(int) ]) 
        ########## method 1:
        randindex = np.round(self.random_number_generator.rand(self.pop_size) * nTop).astype(int)  # select from [1, 2, 3, ..., pNP]
        sorted_population =  sorted(zip(self.population, self.population_energies), key = lambda  x: x[1])
        _max = max(randindex)
        _max_sorted_population = np.array([sorted_population[x][0].tolist() for x in range(_max+1)])
        pbest = np.array([_max_sorted_population[x] for x in randindex])
        ######### method 2:
        #sorted_index = np.array( sorted(enumerate(self.population_energies), key = lambda  x: x[1]) )[:,0].astype(int) 
        # _si = np.zeros((self.pop_size))
        # for i in range(self.pop_size):
        #   _si[sorted_index[i]] = i
        # _si = np.array([list(range(self.pop_size)), list(sorted_index)])[:,[1,0]]
        #pbest = self.population[ _si[randindex] ]

        ## computing the random r1 and r2 vectors        
        r1, r2 = self._genR1R2(self.pop_size, self.pop_size + len(self.archive_pop))
        
        ### mutation
        vi = self.population + \
                np.tile(sf, (self.parameter_count,1)).transpose() * \
                (pbest - self.population + self.population[r1] - np.concatenate((self.population, self.archive_pop))[r2])
        
        ### crossover
        #ui = vi
        jmask = ( self.random_number_generator.uniform(0,self.parameter_count,(self.pop_size, self.parameter_count)).astype(int) == np.tile(np.array(range(self.parameter_count)),(self.pop_size,1)) )
        mask = jmask & ( self.random_number_generator.rand(self.pop_size, self.parameter_count) > np.tile(cr, (self.parameter_count,1)).transpose() )
        vi[mask] = self.population[mask]
        
        ## computing prediction
        for trial in vi: self._ensure_constraint(trial)
        parameters = np.array([self._scale_parameters(trial) for trial in vi])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize

        # Update archieve
        better_child_indx = (self.population_energies > energies)
        goodCR, goodF = np.zeros((self.pop_size)), np.zeros((self.pop_size))
        goodCR[better_child_indx] = cr[better_child_indx]
        goodF[better_child_indx] = sf[better_child_indx]
        self._update_archive(better_child_indx, energies)   

        # update cr and f in history memeory
        abs_dif_fitness = abs(energies[better_child_indx] - self.population_energies[better_child_indx])        
        if sum(better_child_indx) > 0:   
            self._update_memory_f_cr(abs_dif_fitness, goodF[better_child_indx], goodCR[better_child_indx])

            ## selecting and update self.population_energies   
            self.population[better_child_indx] = vi[better_child_indx]
            self.population_energies[better_child_indx] = energies[better_child_indx] 
            
            # exchange 0 and min_pos, for consistence of the parent implementation
            min_pos = np.argmin(self.population_energies)
            if min_pos != 0:
                t1, t2 = self.population_energies[min_pos], np.array( self.population[min_pos] )
                '''
                One awful bug: the original above line is:
                t1, t2 = self.population_energies[min_pos],  self.population[min_pos] 
                It does not exchange self.population[0] with self.population[min_pos].
                In deed, it leads to self.population[0] == self.population[min_pos].
                -YS, 2020.09.19
                '''
                self.population_energies[min_pos], self.population[min_pos]  = self.population_energies[0], self.population[0]
                self.population_energies[0], self.population[0] = t1, t2 

        return self.x, self.population_energies[0]

    def _update_archive(self, beter_child_indx, energies):

        self.archive_pop = np.concatenate((self.archive_pop, self.population[beter_child_indx]))
        self.archive_funvalues = np.concatenate((self.archive_funvalues, energies[beter_child_indx]))
        self.archive_pop, indx = np.unique(self.archive_pop, return_index=True, axis=0)
        self.archive_funvalues = self.archive_funvalues[indx]

        if len(self.archive_pop) > self.archive_NP:
           pos = np.random.permutation(np.array(range(len(self.archive_pop))))
           pos = pos[0:self.archive_NP-1]
           self.archive_pop = self.archive_pop[pos]
           self.archive_funvalues = self.archive_funvalues[pos]

    def _update_memory_f_cr(self, abs_dif_fitness, goodF, goodCR):
        wi = abs_dif_fitness / sum(abs_dif_fitness)
        self.memory_f[self.memory_pos] = sum((wi * (goodF**2))) / sum(wi * goodF)
        if max(goodCR) == 0:
            self.memory_cr[self.memory_pos]  = -1
        else:
            self.memory_cr[self.memory_pos] = sum((wi * (goodCR ** 2))) / sum(wi * goodCR)

        self.memory_pos = (self.memory_pos + 1) % self.memory_size

    def _genR1R2(self, size0, size1):  
        # generate r1 and r2 from [range(size0)] and [range(size1)] respectively such that
        # r1[i] !=  r0[i], r2[i] != r1[i] and r2[i] != r0[i] for every i
        # where r0 = [range(self.pop_size)]
        r0 = np.array(range(self.pop_size))
        r1 = (self.random_number_generator.rand(size0) * size0).astype(int)
        pos = (r0 == r1)
        while( sum(pos) != 0):
            r1[pos] = (self.random_number_generator.rand(sum(pos)) * size0).astype(int)
            pos = (r0 == r1)

        r2 = (self.random_number_generator.rand(size0) * size1).astype(int)
        pos = ((r2 == r1) | (r2 == r0))
        while( sum(pos) != 0):
            r2[pos] = (self.random_number_generator.rand(sum(pos)) * size1).astype(int)
            pos = ((r2 == r1) | (r2 == r0))
            
        return r1, r2#.astype(int), r2.astype(int)
            
class EBLSHADE(SHADE):
    # First_calss_percentage=0.5
    MCF=0.5 
    min_pop_size=4
    L_Rate= 0.80
    max_nfes = 30000.0
    def __init__(self, func, bounds, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0, iter_LS=5, no_LS=True, LS=0,
                 memory_size=100, p_best_rate=0.11, arc_rate=1.4, EDE_best_rate=0.1): 
        super().__init__(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback,
                                         disp=disp, init=init, atol=atol, LS=LS,
                                         memory_size=memory_size, p_best_rate=p_best_rate, arc_rate=arc_rate)            
        # Class#1 probability for Hybridization                                              
        self.memory_FCP = (np.ones((memory_size)) * self.MCF).astype(np.float32)
        self.EDE_best_rate = EDE_best_rate
        self.max_pop_size = self.pop_size #* len(bounds)

    def _genR(self, size): # generate random 2 vectors of range(size)
        r1 = (self.random_number_generator.rand(size) * size).astype(int)
        r2 = (self.random_number_generator.rand(size) * size).astype(int)
        pos = (r1 == r2)
        while( sum(pos) != 0):
            r2[pos] = (self.random_number_generator.rand(sum(pos)) * size).astype(int)
            pos = (r2 == r1)
        return r1, r2

    def _update_memory_f_cr_fcp(self,abs_dif_fitness, goodF, goodCR, dif_val_Class_1, dif_val_Class_2):        
        _sum_class = sum(dif_val_Class_1)+sum(dif_val_Class_2)
        if _sum_class > 0:
            self.memory_FCP[self.memory_pos] = self.memory_FCP[self.memory_pos] * self.L_Rate +\
                (1-self.L_Rate)*sum(dif_val_Class_1)/_sum_class
        self.memory_FCP[self.memory_pos] = max(0.2, min(self.memory_FCP[self.memory_pos], 0.8))
        self._update_memory_f_cr(abs_dif_fitness, goodF, goodCR)

    def _resize_population(self, sorted_index):        
        plan_pop_size = abs(int((((self.min_pop_size - self.max_pop_size) / self.max_nfes) * self._nfev) + self.max_pop_size))
        plan_pop_size = max(self.min_pop_size, plan_pop_size)

        if self.pop_size > plan_pop_size:
            #reduct_size = plan_pop_size - self.pop_size
            best_index = sorted_index < plan_pop_size
            self.population = self.population[best_index]
            self.population_energies = self.population_energies[best_index]
            self.population_shape = self.population.shape 
            self.pop_size = plan_pop_size
            self.archive_NP = self.pop_size * self.arc_rate
            if self.archive_NP < len(self.archive_pop):
                ### THE BELOW LINE IS ORIGIANLLY AS 
		## self.archive_pop =  np.random.permutation(range(self.archive_NP))[:self.archive_NP] 
		### WHICH SHOULD BE CORRECTED AS THE BELOW LINE 
                self.archive_pop = self.archive_pop[ np.random.permutation(range(self.archive_NP))[:self.archive_NP] ]

    def __next__(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()
        """
        if self.dither is not None:
            self.scale = (self.random_number_generator.rand()
                          * (self.dither[1] - self.dither[0]) + self.dither[0]) 
        """
        itersize = max(0, min(self.num_population_members, self.maxfun - self._nfev + 1))

        #### mutation and crossover
        mem_rand_index = (self.memory_size * self.random_number_generator.rand(self.pop_size)).astype(int)

        mu_sf = self.memory_f[mem_rand_index]
        mu_cr = self.memory_cr[mem_rand_index]
        class_select_index = ( self.memory_FCP[ mem_rand_index ] >= self.random_number_generator.rand(self.pop_size))

        cr = self.random_number_generator.normal(mu_cr, 0.1)
        cr[ cr < 0 ] = 0.
        cr[ cr > 1 ] = 1.

        sf = mu_sf + 0.1 * np.tan(np.pi * (self.random_number_generator.rand(self.pop_size) - 0.5))
        sf[ sf > 1 ] = 1.
        pos = np.where(sf <= 0)[0]
        while np.size(pos) > 0:
            sf[pos] = mu_sf[pos] + 0.1 * np.tan(np.pi * (self.random_number_generator.rand(np.size(pos)) - 0.5))
            pos = np.where(sf <= 0)[0]

        ## computing the pbest vectors
        p_best_rate = self.random_number_generator.uniform(2./self.pop_size, 0.2, self.pop_size)
        nTop = np.array([max(2, x) for x in (p_best_rate * self.pop_size).round().astype(int) ])        
        randindex = np.round(self.random_number_generator.rand(self.pop_size) * nTop).astype(int)  # select from [1, 2, 3, ..., pNP]
        sorted_index = np.array( sorted(enumerate(self.population_energies), key = lambda  x: x[1]) )[:,0].astype(int) 
        _swap_sorted_index = np.zeros((self.pop_size)).astype(int)
        for i in range(self.pop_size):  _swap_sorted_index[sorted_index[i]] = i 
        #_swap_sorted_index.astype(int)
        #sorted_population = np.array(list( np.array( sorted(zip(self.population, self.population_energies), key = lambda  x: x[ 1]) ) ))[:,0]
        pbest = self.population[ _swap_sorted_index[randindex] ]

        ## computing the random r1 and r2 vectors        
        r1, r2 = self._genR1R2(self.pop_size, self.pop_size + len(self.archive_pop))
        
        ### mutation
        vi = self.random_number_generator.random_sample(self.population_shape)
        ## in terms of p_best
        if sum(class_select_index) != 0:
            vi[class_select_index] = self.population[class_select_index] + \
                    np.tile(sf, (self.parameter_count,1)).transpose()[class_select_index] * \
                    ( pbest[class_select_index] - self.population[class_select_index] + \
                    self.population[r1[class_select_index]] - np.concatenate((self.population, self.archive_pop))[r2[class_select_index]])
        ## in terms of ord_best
        if sum(~class_select_index) != 0:
        #  select top EDE_best_rate% (10%) to be inculded in EDE 
            EDEpNP = max(2, int(self.EDE_best_rate * self.pop_size)) # choose at least two best solutions
            EDErandindex = (self.random_number_generator.rand(self.pop_size) * EDEpNP).astype(int)  #select from [1, 2, 3, ..., pNP] 
            EDEpestind=_swap_sorted_index[ EDErandindex ] # the pbest index
            EDE_r1_ind, EDE_r2_ind = self._genR(self.pop_size)
            r_b, r_w, r_m = np.zeros((self.pop_size)).astype(int), np.zeros((self.pop_size)).astype(int), np.zeros((self.pop_size)).astype(int)
            for i in range(self.pop_size):
                [r_b[i], r_m[i], r_w[i]] = np.array( sorted([[0, self.population_energies[EDEpestind[i]]], \
                    [1, self.population_energies[EDE_r1_ind[i]]], \
                    [2, self.population_energies[EDE_r2_ind[i]]]], \
                    key=lambda x: x[1]) )[:,0].astype(int) 

            class_select_index = ~class_select_index
            vi[class_select_index] = self.population[class_select_index] + \
                    np.tile(sf, (self.parameter_count,1)).transpose()[class_select_index] * \
                    ( self.population[ r_b[class_select_index] ]  - self.population[class_select_index] + \
                    self.population[r_m[class_select_index]] - np.concatenate((self.population, self.archive_pop))[r_w[class_select_index]])
        ### crossover
        #ui = vi
        jmask = ( self.random_number_generator.uniform(0,self.parameter_count,(self.pop_size, self.parameter_count)).astype(int) == np.tile(np.array(range(self.parameter_count)),(self.pop_size,1)) )
        mask = jmask & ( self.random_number_generator.rand(self.pop_size, self.parameter_count) > np.tile(cr, (self.parameter_count,1)).transpose() )
        vi[mask] = self.population[mask]
        
        ## computing prediction
        for trial in vi: self._ensure_constraint(trial)
        parameters = np.array([self._scale_parameters(trial) for trial in vi])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize

        # Update archieve
        better_child_indx = (self.population_energies > energies)
        goodCR, goodF = np.zeros((self.pop_size)), np.zeros((self.pop_size))
        goodCR[better_child_indx] = cr[better_child_indx]
        goodF[better_child_indx] = sf[better_child_indx]
        self._update_archive(better_child_indx, energies)   
        abs_dif_fitness = np.zeros((self.pop_size))
        if sum(better_child_indx) > 0:   
            # update cr and f in history memeory
            abs_dif_fitness[better_child_indx] = abs(energies[better_child_indx] - self.population_energies[better_child_indx])        
            # update the FCP
            dif_val_class_1 = abs_dif_fitness[ better_child_indx & ~class_select_index]
            dif_val_class_2 = abs_dif_fitness[ better_child_indx & class_select_index]
        
            self._update_memory_f_cr_fcp(abs_dif_fitness, goodF, goodCR, dif_val_class_1, dif_val_class_2) 

            ## selecting and update self.population_energies   
            self.population[better_child_indx] = vi[better_child_indx]
            self.population_energies[better_child_indx] = energies[better_child_indx] 
        
            # exchange 0 and min_pos, for consistence of the parent implementation
            min_pos = np.argmin(self.population_energies)
            if min_pos != 0:
                t1, t2 = self.population_energies[min_pos], np.array( self.population[min_pos] ) 
                self.population_energies[min_pos], self.population[min_pos]  = self.population_energies[0], self.population[0]
                self.population_energies[0], self.population[0] = t1, t2 

        ## resizing the population size
        self._resize_population(_swap_sorted_index)        

        return self.x, self.population_energies[0]
