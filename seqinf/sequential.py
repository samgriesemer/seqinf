import logging
from collections import defaultdict
from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor, nn
from torch.distributions import Distribution
from tqdm.auto import tqdm

import sbi
from sbi.utils import (
    get_density_thresholder,
    RestrictedPrior
)
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.utils.sbiutils import seed_all_backends
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.torchutils import process_device

import sbibm
from sbibm.tasks import Task

from seqinf.util import prepare_sbibm_prior
from seqinf.collector import Collector


logger = logging.getLogger(__name__)


class SequentialInference(ABC):
    '''
    Multi-round simulation-based inference wrapper class

    Serves as a general constructor for ``NeuralInference``-based estimators.
    In the ``sbi`` package, all of the following classes inherit from the
    ``NeuralInference`` base class:

    - Likelihood-based (``LikelihoodEstimator``)
        - ``NLE_A``: [Sequential NLE][1]
        - ``MNLE``: [Mixed NLE][2]
    - Posterior-based (``PosteriorEstimator``)
        - ``NPE_A``: [SNPE 2016][3]
        - ``NPE_B``: [SNPE-B][4]
        - ``NPE_C``: [APT][5]
    - Likelihood ratio-based (``RatioEstimator``)
        - ``NRE_A``: [AALR][6]
        - ``NRE_B``: [SRE][7]
        - ``NRE_C``: [Contrastive NRE][8]
        - ``BNRE``: [Balanced NRE][9]
    - Posterior score-based (``NPSE``)
        - ``NPSE`` (no current multi-round support)
            - [SNSE][10]
            - [Score modeling for SDBI][11]
    - Posterior flow-matching (``FMPE``)
        - ``FMPE``: [Flow matching][12]

    NOTE: the typical class hierarchy in the ``sbi`` package looks something
    like

    .. code-block:: sh

        NeuralInference [general wrapper base]
         | -> get_simulations()
         | -> append_simulations()
         | -> train()
         |
        LikelihoodEstimator [intermediate method-specific class]
         | -> build_posterior()
         |     |
         |     |    NeuralPosterior [general posterior sampler base]
         |     |     | -> sample()
         |     |     | -> potential()
         |     |     |
         |     | -> MCMCPosterior [concrete sampling method-specific class]
         |
        NLE_A [concrete approach]


    These show up in the following methods as:

    - ``method_func`` returns an instanced of ``NeuralInference``; will be a
      particular subclass based on the provided ``method`` string (e.g.,
      ``PosteriorEstimator`` for ``"NPE_C"``)
    - ``method_func`` wrappers (e.g., high-level ``sbi`` objects like ``SNPE``
      or ``NLE_A``) can accept a string for the density estimator model (which
      will handle initialization of the model automatically), or you can pass a
      builder function that will return a constructed model when invoked. The
      latter case is how we enable custom DE properties, such as hyperparameter
      specification (e.g., dropout), using an embedding net, etc. There are
      helper methods that *return* the needed builder function, and we can pass
      our elements to these helpers and let them take care of *preparing the
      setup*, to later be invoked when ``method_func`` is initialized.
    - ``_build_posterior()`` produces an instance of ``NeuralPosterior``,
      matching the requested means of sampling as given by ``sample_with``
      (e.g., ``DirectPosterior`` for ``sample_with="direct"``,
      ``RejectionPosterior`` for ``sample_with="rejection"``). The outer
      method, as indicated by ``method``, affects the options available here.


    [1]: https://arxiv.org/abs/1805.07226
    [2]: https://www.biorxiv.org/content/10.1101/2021.12.22.473472v2
    [3]: https://arxiv.org/abs/1605.06376
    [4]: https://arxiv.org/abs/1711.01861
    [5]: https://arxiv.org/abs/1905.07488
    [6]: https://arxiv.org/abs/1903.04057
    [7]: https://arxiv.org/pdf/2002.03712
    [8]: https://arxiv.org/abs/2210.06170
    [9]: https://arxiv.org/abs/2208.13624
    [10]: https://arxiv.org/abs/2210.04872
    [11]: https://arxiv.org/abs/2209.14249
    [12]: https://arxiv.org/abs/2305.17161
    '''
    collector_cls: type[Collector] = Collector

    def __init__(
        self,
        method: str | Callable,
        simulator: Callable,
        prior: Distribution | None = None,
        density_estimator: str | Callable | None = None,
        de_kwargs: dict | None = None,
        embedding_net: nn.Module = nn.Identity(),

        sample_with: str | None = None,
        sample_method: str | None = None,
        sample_parameters: dict | None = None,
        proposal_method: str | None = None,

        device: str = 'cpu',
        num_workers: int | None = None,
        retrain_from_scratch: bool = False,

        task: Task | None = None
    ):
        '''
        Set up multi-round inference objects.

        Constructs the base ``NeuralInference`` object (from the ``sbi``
        package) based on the provided ``method`` and ``density_estimator``,
        and prepares the ``simulator`` and ``prior``.

        .. admonition:: Preparing the simulator

            When it comes to preparing the simulator, ``process_simulator()``
            attempts to check for native batch support within the passed
            callable. That is, it expects the function to be able to handle
            tensors with a batch dimension such that, if N is the parameter
            size and M is the output size, it can map from sizes ``(B, N) ->
            (B, M)``.

            It also appears that, as of v0.23, simulators are only ever passed
            tensors (although previously there was a ``is_numpy_simulator``
            flag). Nevertheless, while tensors are cast to Numpy arrays for
            more efficient batch handling with ``joblib``, they should more
            than likely be written to expect and return a ``torch.Tensor`` in
            the first place.

            If the simulator is not natively vectorizable (e.g., is actually a
            ``subprocess`` call that first unpacks or parses the tensor
            inputs), ``process_simulator`` will wrap it in a decorator that can
            accept a batched tensor. This makes it easy to use with
            ``simulate_in_batches`` (which is now deprecated in ``sbi`` but a
            version is available here in ``seqinf``), as workers can pass in
            their batched parameters and have it behave as expected.
            Internally, that decorator will just unpack the batched tensor and
            feed parameter vectors one-by-one to the user-passed function,
            without the batch dimension. The exact trace of utility methods
            that do handle this wrapping is

            .. code-block:: sh

                process_simulator
                -> ensure_batched_simulator
                -> get_batch_loop_simulator

            where ``get_batch_loop_simulator`` returns a wrapped simulator that
            just calls the simulator in a loop for each parameter but accepts a
            batched tensor input.

        Misc remarks:

        - The "method" objects (e.g., ``sbi.inference.NPE``) are mostly density
          estimator wrappers (often supporting many different NDE options).
          They directly expose a ``.train()`` method which can be called after
          adding samples with ``.append_simulations()``. They also natively
          store round-wise data, retrievable with ``.get_simulations()``.
        - For ``sample_*`` arguments, see more details in
          ``_build_proposal()``.

        Parameters:
            method: NeuralInference-based method reference
            simulator: callable simulator
            prior: prior distribution over parameters
            density_estimator: NDE model architecture to use when learning the
                relevant probabilistic form
            embedding_net:
                embedding network to use for embedding the output
            sample_with: means of sampling the posterior
            sample_method: specific method variant of ``sample_with``, if
                applicable
            sample_parameters: additional parameters for posterior sampler
            proposal_method: method for constructing round-wise proposals
            device: PyTorch device string to use for training, sampling, etc.
                Default is "cpu," and in most cases changing to GPU (setting to
                "cuda") will not yield a speed-up (according to ``sbi`` docs).
            num_workers: number of workers to use for batch processing
                simulation outputs
            retrain_from_scratch: whether or not to retrain the NDE only using
                the samples drawn each round
            task: SBI-BM task object if initialized from a task context
        '''
        try:
            if type(method) is str:
                self.method_func: Callable = getattr(sbi.inference, method.upper())
            else:
                self.method_func = method
        except AttributeError:
            raise NameError(
                f'SBI method "{method}" not available, please use a supported alias'
            )

        self.density_estimator = density_estimator
        self.de_kwargs = de_kwargs if de_kwargs else {}
        self.embedding_net = embedding_net

        self.sample_with = sample_with
        self.sample_method = sample_method
        self.sample_parameters = sample_parameters
        self.proposal_method = proposal_method

        self.device = process_device(device)
        self.num_workers = num_workers
        self.retrain_from_scratch = retrain_from_scratch

        self.task = task

        # computed
        self.prior, _, is_np = process_prior(prior)
        self.simulator = process_simulator(simulator, self.prior, is_np)
        self.collector = self.collector_cls(self.simulator, num_workers)

        self._seed = None
        self._seedwise_inference = {}
        self._seedwise_xs = defaultdict(list)
        self._seedwise_thetas = defaultdict(list)
        self._seedwise_proposals = defaultdict(list)

        # only relevant if SBIBM task provided
        self._seedwise_num_observation = {}

        check_sbi_inputs(self.simulator, self.prior)
        logger.debug('Simulator and prior verified')

        # additional parsing
        self._set_sample_args()
        self._reset()

    @classmethod
    def from_task(
        cls,
        method: str,
        task_name: str,
        **kwargs,
    ):
        task = sbibm.get_task(task_name)
        assert task is not None

        simulator = task.get_simulator()
        prior = task.prior_dist

        return cls(
            method,
            simulator=simulator,
            prior=prior,
            task=task,
            **kwargs
        )

    def _set_sample_args(self):
        '''
        Set the sample kwargs dict for ``build_posterior(...)``.

        This is designed around the ``.build_posterior()`` methods of various
        methods. The options available do vary by method; for instance,
        ``sample_with="direct"`` is only available for direct posterior methods
        like SNPE. This means there can be issues downstream if one tries to
        set options that aren't actually available for the method, but we
        assume the user is aware of these options (see the ``sbi`` API or some
        nominal details in the ``.build_proposal()`` method of this class).

        Basically: only set ``sample_method`` and ``sample_parameters`` if you
        know they can be used with the provided ``sample_with``.
        '''
        self.sample_kwargs = {}

        # set up keys
        sample_method_key = f'{self.sample_with}_method'

        sample_with = self.sample_with
        if sample_with == 'rejection' or sample_with == 'importance':
            sample_with = f'{sample_with}_sampling'

        sample_params_key = f'{sample_with}_parameters'

        # set arguments
        if self.sample_with:
            self.sample_kwargs['sample_with'] = self.sample_with

        # only options for VI and MCMC
        if self.sample_method:
            self.sample_kwargs[sample_method_key] = self.sample_method

        if self.sample_parameters:
            self.sample_kwargs[sample_params_key] = self.sample_parameters

    @abstractmethod
    def _density_builder_wrapper(self) -> Callable:
        '''
        Abstract density estimator building step.

        Inheriting classes are made concrete by defining this method. Because
        we allow for customization of the density estimator construction, this
        step is not automatically handled by the outer ``method_func`` call.
        Instead, this method should very likely call an appropriate helper from
        ``sbi.neural_nets``, like ``posterior_nn``. For instance,

        .. code-block:: python

            return posterior_nn(
                model=self.density_estimator,
                embedding_net=self.embedding_net,
                <child-specific kwargs>,
                **self.de_kwargs,
            )

        This method is then invoked when setting up the inference wrapper, and
        should itself return a callable. That is, this method serves as a
        *wrapper* of a *density building function* to later be called when
        actually setting up the density estimator. See ``_reset()``.

        NOTE: while we could perhaps infer the intended neural network
        "direction" based on the outer method (e.g., using ``posterior_nn``
        when the method is NPE-based), we use this as a meaningful place to
        break the ``SequentialInference`` into abstract territory. Because
        we're so general in the setup, e.g., allowing one constructor for SNPE,
        SNLE, SNRE, etc, it's useful to require concrete instances "take a
        beat" and consider the differences between methods. Outside the
        inference procedure itself, this so happens to be one of the clear
        places where each method can differ heavily; for instance, NPE_A
        requires an MDN while NPE_C can use flows, SPNE works on score-based
        models, etc. The helpers like ``posterior_nn`` and ``likelihood_nn``
        also have different arguments, so it isn't particularly easy to
        streamline this setup step without heavily restricting customization.
        '''
        ...

    def _reset(self):
        '''
        Reset the actively updated round-wise instance variables.

        Recreates the inference object from the original arguments, wiping all
        simulations and round-wise stores. Resets the proposal back to the
        prior.

        NOTE: we generally don't expect to call this method with the same seed
        twice, given we check (in places like ``run()``) to see if we ran
        inference for a particular seed and exit early if so. We store run data
        with dicts that use seeds as keys, and we clear the seeded entries in
        these stores for the passed seed. So we're almost always calling this
        method with a unique seed, which means we aren't usually clearing any
        of that seeded metadata and only resetting the instance globals.

        NOTE: this method is called from the constructor after explicitly
        setting ``self._seed=None``. This has the effect of creating a "dud
        entry" given we add the prior as the first proposal in each case. Given
        we always generate a concrete seed elsewhere, this should be the only
        bit of seeded metadata assigned to ``None``. This makes it easy for use
        to use the prior as a proposal in other streamlined access contexts
        (e.g., in ``InferenceDiagnostic``), i.e., we can access the prior with
        ``self.get_round_proposal()`` without needing to have first called
        ``run()``.
        '''
        seed_all_backends(self._seed)

        self._seedwise_xs.pop(self._seed, None)
        self._seedwise_thetas.pop(self._seed, None)
        self._seedwise_proposals.pop(self._seed, None)
        self._seedwise_inference.pop(self._seed, None)

        self.proposal = self.prior
        self._seedwise_proposals[self._seed].append(self.proposal)

        self.trained_de = None
        self.inference = self.method_func(
            prior=self.prior,
            density_estimator=self._density_builder_wrapper(),
            device=self.device,
        )

    def _build_proposal(self, x_o: Tensor = None):
        '''
        Build a posterior from a (trained) density estimator in the context of
        the SBI method.

        Most of the differences in how such a posterior model is constructed
        are abstracted away by the ``sbi`` package through
        ``<method>.build_posterior()``. We can nevertheless still influence a
        few behavioral components through this method, primarily through the
        ``sample_with`` parameter.

        The value of ``sample_with`` indicates the sampling strategy used when
        drawing from the posterior model. It can be any of the following:

        - Direct (``direct``): SNPE and FMPE * Wrapper class:
          ``DirectPosterior`` * Draws samples directly from the posterior NDE
          (again, NPE only), and corrects for leakage outside the prior bounds
          (rejects samples outside and corrects log probs accordingly) * Other
          parameters: + ``enable_transform``: defaults to true, will transform
          parameters to unconstrained space during MAP optimization.
        - Markov Chain Monte Carlo (``mcmc``): SNPE, SNLE, SNRE * Wrapper
          class: ``MCMCPosterior`` * Samples from the posterior with the
          forward model using MCMC
        - Rejection sampling (``rejection``): SNPE, SNLE, SNRE * Wrapper class:
          ``RejectionPosterior`` * Samples from the posterior with the forward
          model using rejection sampling
        - Importance samples (``importance``): SNPE, SNLE, SNRE * Wrapper
          class: ``ImportanceSamplingPosterior`` * Samples from the posterior
          with the forward model using importance sampling
        - Variational inference (``vi``): SNPE, SNLE, SNRE * Wrapper class:
          ``VIPosterior`` * Learns a tractable variational posterior
          distribution q(θ) (optimizes the variational params to minimize the
          KL divergence between q and the posterior)

        Note that the last 4 methods are more aimed at SNLE and SNRE methods,
        as they don't learn a direct posterior model. In these cases, the
        method operates on a "potential" (unnormalized log prob, e.g.,
        p(x|θ)p(θ) rather than p(θ|x)). The ``.log_prob()`` is thus only
        available in the variational inference case; in VI we get a new,
        tractable distribution we can use directly, but we otherwise only have
        log-probs up to the normalizing constant (and the method is actually
        explicitly deprecated in these cases; if needed, use ``.potential()``
        instead). 

        Specifically in score-based contexts, we also have a
        ``ScorePosterior``, which constructs a posterior that can be sampled
        through the learned diffusion model (and enables true ``.log_prob()``).

        To enable the potential-based posterior samplers, we have method-wise
        potentials: posterior/likelihood/ratio/score-based classes inheriting
        from ``BasePotential``. It's worth noting we don't really need this in
        the posterior case (i.e., when learning the posterior directly) since
        we can use the true probs directly, and could technically pass those
        into the potential-driven samplers even if we wanted to use them.
        Nevertheless, a potential class still exists in this case, but its

        > "potential is the same as the log-probability of the
        `posterior_estimator`, but it > is set to $-\inf$ outside of the prior
        bounds."

        So we don't actually have to do much extra with the output of the model
        (whereas in the likelihood case we need to also consider the prior;
        again, p(x|θ)p(θ) vs p(θ|x)). But because we otherwise **don't**
        explicitly consider the prior after training our network, we at some
        point do need to correct for possible leakage outside of prior bounds
        (alluded to above under "Direct"). This is where that happens: we
        assign log-probs outside the prior bounds to $-\inf$ when we use this
        posterior-potential wrapper, which again doesn't actually change the
        probabilities beyond this effective "clipping" (i.e., we don't reduce
        it to an unnormalized potential somehow). The ``NeuralPosterior`` base
        class in general expects a ``potential_fn`` anyhow; we do this
        correction inside the ``DirectPosterior`` class before calling
        ``super().__init__`` and "passing up" the clipped potential.

        Additionally worth noting that ``NeuralPosteriors``, in addition to the
        potential, accept a ``theta_transform``. This is a ``torch.Transform``
        that maps (mostly) bounded prior domains (e.g., box uniforms) to
        unconstrained spaces (e.g., the reals) for more efficient optimization
        (e.g., for finding the posterior MAP, where we use gradient ascent of
        the potential in the transformed, unconstrained parameter space to find
        parameters with the highest posterior likelihood and then map it back
        to the bounded domain). In some cases, I've had issues with this
        transform (I think mapping back to NaNs or something), and it can be
        disabled as seen above with the ``Direct`` parameters
        (``DirectPosterior`` actually expects a full "posterior estimator"
        rather than a potential function, and the transform is computed when
        ``DirectPosterior`` is initialized given we actually call
        ``posterior_estimator_based_potential`` internally). For other
        posterior wrappers, we expect the potential and transform to have been
        defined externally and passed in.

        More details on the relevant instance variables (expected in
        ``__init__()``):
            
        - sample_with: means of sampling the posterior; see the keyword table
          above
        - sample_method: specific method variant of ``sample_with``, if
          applicable (e.g., ``mcmc_method`` can use different MCMC variants
          such as ``slice``, ``hmc``, ``nuts``, etc)
        - sample_parameters: additional parameters to use when constructing the
          concrete ``NeuralPosterior`` (e.g., ``enable_transform=False`` in
          ``DirectPosterior``, so should be coupled with
          ``sample_with="direct"``)
        - proposal_method: method for constructing round-wise proposals. This
          can be "posterior" (default) to construct and/or sample from the
          posterior, or "truncated" to sample from the truncated prior (i.e.,
          TSNPE).

        Parameters:
            x_o: observational data point

        Returns: new proposal (posterior sampler conditioned on x_o and
            adhering to the proposal method)
        '''
        # produce posterior sampler from NDE
        posterior: NeuralPosterior = self.inference.build_posterior(
            density_estimator=self.trained_de,
            prior=self.prior,
            **self.sample_kwargs
        )

        if x_o is not None:
            posterior = posterior.set_default_x(x_o)

        return self._apply_proposal_method(posterior)

    def _apply_proposal_method(self, posterior: NeuralPosterior):
        # set next round proposal
        if self.proposal_method == 'posterior':
            proposal = posterior
        elif self.proposal_method == 'truncated':
            accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
            proposal = RestrictedPrior(
                self.prior,
                accept_reject_fn,
                sample_with="rejection"
            )

        return proposal

    def _train_de(self, theta: Tensor, x: Tensor):
        '''
        Trains the density estimator using provided simulator samples.

        Parameters:
            theta: simulation parameters, as a tensor shaped (B, N)
            x: simulation outputs corresponding to the parameters, as a tensor
                shaped (B, M)
        '''
        # only SNPE should all proposal, and we ignore in TSNPE case
        allows_proposal = (
            'proposal' in self.inference.append_simulations.__code__.co_varnames
        )

        # handle TSNPE-specific setting
        if self.proposal_method == 'truncated':
            trained_de = self.inference.append_simulations(
                theta, x,
            ).train(
                force_first_round_loss=True,
                retrain_from_scratch=self.retrain_from_scratch,
            )
        else:
            if allows_proposal:
                prepped_de = self.inference.append_simulations(
                    theta, x, proposal=self.proposal
                )
            else:
                prepped_de = self.inference.append_simulations(
                    theta, x
                )

            trained_de = prepped_de.train(
                retrain_from_scratch=self.retrain_from_scratch,
            )

        return trained_de

    def run(
        self,
        num_rounds: int = 1,
        num_samples: int = 100,
        x_o: Tensor | int | None = None,
        seed: int | None = None,
        **collect_kwargs
    ):
        '''
        Run sequential inference under observation ``x_o``.

        First resets instance variables that change round-wise, then
        iteratively does the following:

        1. **Collects new samples**: parameters are drawn from the current
            proposal, optionally refined or filtered based on the ``Collector``
            implementation, and ran through the simulator.
        2. **Updates the NDE**: new samples are used to train the density
            estimator, optionally according to first round loss (if proposals
            are truncated priors, as with TSNPE) or from scratch each round.
        3. **Builds the next round's proposal**: the trained NDE is wrapped in
            a posterior "sampler" and conditioned on the observational data
            $x_o$. In some cases (non-NPE), we only have a posterior potential,
            but can sample via MCMC/rejection/importance/VI, and we do so
            according to the provided ``sample_*`` constructor arguments.

        Parameters:
            num_rounds: number of rounds to perform inference over
            num_samples: number of samples to draw/simulate each round
            x_o: observational data to condition on round-to-round; can be a
                raw torch.Tensor or an integer representing an SBIBM
                observation number if ``self.task`` is defined.
            seed: random seed to set prior to running. If None, one will be
                generated and explicitly set to store run metadata.
        '''
        if seed is None:
            seed = np.random.randint(2**31)

        if seed in self._seedwise_inference:
            logger.info(f'Inference loop already ran for seed {seed}')
            return

        # the order is important here
        self._seed = seed
        self._reset()
        self._seedwise_inference[self._seed] = self.inference

        if type(x_o) is int and self.task is not None:
            logger.info(f'Treating `x_o` value {x_o} as SBIBM task observation num')

            num_observation = x_o
            self._seedwise_num_observation[self._seed] = num_observation
            x_o = self.task.get_observation(num_observation=num_observation)

        for i in tqdm(range(num_rounds), desc='Rounds'):
            theta, x = self.collector.collect(
                proposal=self.proposal,
                num_samples=num_samples,
                seed=self._seed,
                **collect_kwargs
            )

            self.trained_de = self._train_de(theta, x)
            self.proposal = self._build_proposal(x_o)

            self._seedwise_xs[self._seed].append(x)
            self._seedwise_thetas[self._seed].append(theta)
            self._seedwise_proposals[self._seed].append(self.proposal)

    def get_round_xs(
        self,
        from_round: int | None = -1,
        with_seed: int | None = None
    ):
        if with_seed is None:
            with_seed = self._seed

        return self._seedwise_xs[with_seed][from_round]

    def get_round_thetas(
        self,
        from_round: int | None = -1,
        with_seed: int | None = None
    ):
        if with_seed is None:
            with_seed = self._seed

        return self._seedwise_thetas[with_seed][from_round]

    def get_round_proposal(
        self,
        from_round: int | None = -1,
        with_seed: int | None = None
    ):
        if with_seed is None:
            with_seed = self._seed

        return self._seedwise_proposals[with_seed][from_round]
