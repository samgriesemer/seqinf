import logging
from collections import defaultdict

import torch
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sbi.analysis import pairplot
from sbi.neural_nets.estimators.shape_handling import reshape_to_batch_event
from sbi.utils.metrics import posterior_shrinkage, l2

from sbibm.metrics import c2st, mmd, ksd, median_distance, posterior_mean_error

from seqinf.util import printd
from seqinf.collector import Collector, BayesianCollector
from seqinf.sequential import SequentialInference
from seqinf.posterior import BayesDirectPosterior


logger = logging.getLogger(__name__)

COLORS = [
    # "#85160f", "#673c0c", "#4c4928", "#25532e", "#234485",  # l40
    # "#fea294", "#f1ad6f", "#c6c182", "#94cf9c", "#a2bef2",  # l80

    # blue     red        green      orange     yellow
    "#557ecc", "#dc4234", "#4f925a", "#b76b0c", "#888346",  # l60
    "#8daeeb", "#fa897a", "#80c089", "#e79a51", "#b7b170",  # l75
    "#cedffc", "#ffd2ca", "#beecc4", "#fed5b3", "#e4e1b4",  # l90
]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)


class InferenceDiagnostic:
    def __init__(
        self,
        si: SequentialInference,
        num_workers: int = 1,
        x_o: Tensor | None = None,  # TODO
        theta_o: Tensor | None = None,  # TODO
    ):
        '''
        NOTE: we construct a vanilla ``Collector`` here (rather than using the
        SeqInf object's internal collector, which may have filtering or other
        effects) to just facilitate and standardize regular sampling.

        Parameters:
            si: SequentialInference object upon which diagnostic methods will
                act
            num_workers: number of workers to use when drawing simulation
                samples, when needed. Can be set to -1 to use all available CPU
                cores.
        '''
        self.si = si
        self.collector = Collector(self.si.simulator, num_workers)

        self.x_o = x_o
        self.theta_o = theta_o

        self._proposal_samples = {}
        self._proposal_simulations = {}

        self._seedwise_metrics = defaultdict(lambda: defaultdict(dict))

    def _get_map(
        self,
        proposal,
        proposal_samples: Tensor,
    ):
        '''
        Attempt to get the MAP estimate from the provided proposal.

        For true ``NeuralPosterior`` objects, use the ``.map()`` method
        directly. This method is an internal method for safely attempting to
        access the MAP for any distribution that may show up in a sequential
        inference run, including the prior which does *not* have a MAP. We also
        require explicit samples over which to consider the MAP calculations,
        since this method should only ever be called in contexts where we have
        samples already drawn.
        '''
        if self.collector._is_prior(proposal):
            logger.info(
                'Proposal appears to be a prior, no MAP available'
            )
            return None
        else:
            # compute MAP, or get cached value if already computed
            return proposal.map(init_method=proposal_samples)

    def get_theta_dim(self):
        return self.sample_proposal(from_round=0, num_samples=1).shape[1]

    def get_x_dim(self):
        return self.simulate_proposal(from_round=0, num_samples=1).shape[1]

    def sample_proposal(
        self,
        from_round: int = -1,
        num_samples: int = 10_000,
    ):
        '''
        Sample from the proposal distribution at the provided round.

        Use as an internal method to cache samples from individual proposals.
        '''
        # expand last round
        if from_round == -1:
            from_round = len(self.si._seedwise_proposals[self.si._seed])-1

        if (
            from_round not in self._proposal_samples
            or len(self._proposal_samples[from_round]) < num_samples
        ):
            proposal = self.si.get_round_proposal(from_round)
            if from_round in self._proposal_samples:
                pre_samples = self._proposal_samples[from_round]
                num_to_sample = num_samples - len(pre_samples)
                theta = self.collector.sample(proposal, num_to_sample)
                self._proposal_samples[from_round] = torch.vstack(
                    (pre_samples, theta)
                )
            else:
                self._proposal_samples[from_round] = self.collector.sample(
                    proposal, num_samples
                )

        return self._proposal_samples[from_round][:num_samples]

    def simulate_proposal(
        self,
        from_round: int = -1,
        num_samples: int = 1_000,
    ):
        '''
        Simulate from samples drawn from the proposal distribution at the
        provided round.

        Use as an internal method to cache simulations collected from
        individual proposal samples. This method is linked to
        ``sample_proposal()`` in the following way:

        If 100 simulations were already collected under the proposal N and I
        now want to 200, this method will first ask ``sample_proposal()`` for
        200 proposal samples. If only 100 samples had previously been drawn
        from the proposal, ``sample_proposal()`` will stack 100 additional
        samples onto the same 100 used for the original 100 simulations. In
        this method, we then simulate at only the additional 100 samples for
        indices ``100:200`` in the proposal sample list.
        '''
        # expand last round
        if from_round == -1:
            from_round = len(self.si._seedwise_proposals[self.si._seed])-1

        if (
            from_round not in self._proposal_simulations
            or len(self._proposal_simulations[from_round]) < num_samples
        ):
            proposal = self.si.get_round_proposal(from_round)
            theta = self.sample_proposal(from_round, num_samples)
            if from_round in self._proposal_simulations:
                pre_simulations = self._proposal_simulations[from_round]

                # get samples for total simulation count; keeps sim results
                # tied to same proposal sample inputs
                additional_simulations = self.collector.simulate(
                    theta[len(pre_simulations):]
                )
                self._proposal_simulations[from_round] = torch.vstack(
                    (pre_simulations, additional_simulations)
                )
            else:
                self._proposal_simulations[from_round] = self.collector.simulate(
                    theta,
                )

        return self._proposal_simulations[from_round][:num_samples]

    def print_prior_diagnostics(self):
        '''
        Print diagnostic text for the SequentialInference object.
        '''
        printd(f'''
            Shapes:
              Parameter: {self.get_theta_dim()} dims
              Output:    {self.get_x_dim()} dims
        ''')

    def print_run_diagnostics(
        self,
        x_o: Tensor | None = None,
        theta_o: Tensor | None = None,
        seed: int | None = None,
        eval_final: bool = True,
        num_final_samples: int = 1_000,
        num_to_simulate: int = 100,
        sbibm_num_observation: int | None = None
    ):
        '''
        Print diagnostic for a seeded run.

        Parameters:
            seed: seed to use when accessing inference run. None by default,
                  which accesses the latest run.
            eval_final: whether to evaluate the final proposal produced by the
                inference loop. ``SequentialInference.run()`` does not by
                default sample and simulate from this last proposal; setting
                this to True will be sure evaluate the final posterior.
            num_final_samples: number of samples to draw from the final
                proposal for evaluation
            num_to_simulate: number of parameters from the final proposal to
                run through the simulator. If a smaller number (it is by
                default), we pick the ``num_to_simulate`` highest likelihood
                samples that were drawn from the pool of size
                ``num_final_samples`` to actually simulate.
            sbibm_num_observation: if 
        '''
        if seed is None:
            seed = self.si._seed

        if seed not in self.si._seedwise_proposals:
            raise KeyError(
                f'Seed {seed} not recorded, exiting'
            )

        # get round-wise objects
        xs = self.si._seedwise_xs[seed]
        thetas = self.si._seedwise_thetas[seed]
        proposals = self.si._seedwise_proposals[seed]
        N = len(proposals)

        if x_o is None and hasattr(proposals[-1], 'default_x'):
            x_o = proposals[-1].default_x

        # prep observational data point (if present)
        def pvec(vec=None, maxd=5):
            if vec is None:
                return 'None'

            vec_str = ', '.join(map(
                lambda s: f'{"+" if s >= 0 else ""}{s:.2f}',
                torch.atleast_1d(vec.squeeze())[:maxd].numpy()
            ))

            return f'[ {vec_str} ]'

        # sample and simulate from the final posterior
        if eval_final:
            theta = self.sample_proposal(
                from_round=-1,
                num_samples=num_final_samples
            )

            # at highest likelihood (commenting out for now)
            # final_proposal = proposals[-1]
            #
            # if hasattr(final_proposal, 'potential_fn'):
            #     theta_probs = final_proposal.potential_fn(theta).detach()
            # else:
            #     theta_probs = final_proposal.log_prob(theta).detach()
            #
            # # pick the `num_to_simulate` best theta
            # sort_indices = torch.argsort(theta_probs, dim=0)
            # sorted_thetas = theta[sort_indices]
            # thetas_to_simulate = sorted_thetas[-num_to_simulate:]

            # x = self.collector.simulate(thetas_to_simulate)

            x = self.simulate_proposal(
                from_round=-1,
                num_samples=num_to_simulate,
            )

            xs = xs + [x]
            thetas = thetas + [theta]

        printd(f'''
            Roundwise statistics:
            (note: round 1/{N} is the prior, round {N}/{N} is final posterior)

            Observation target (showing at most 5 dims):
               {pvec(x_o)}
        ''')

        # check if SBIBM-based run
        is_sbibm_run = False
        if self.si.task is not None:
            if seed not in self.si._seedwise_num_observation:
                logger.info(
                    f'SBIBM task found, but run "{seed}" has no stored '
                    '`num_observation`. Treating as non-SBIBM run.'
                )
            else:
                is_sbibm_run = True

        prior_samples = thetas[0]
        for i, (x, theta) in tqdm(
            list(enumerate(zip(xs, thetas))),
            desc='Evaluating rounds...',
            leave=False,
        ):
            proposal = proposals[i]
            _map = self._get_map(proposal, theta)

            printd(f'''

                ├──────────────────────────────────────────────────────────────
                │ Round {i+1}/{N}
                ├──────────────────────────────────────────────────────────────
                │
                ├─ General
                │
                │      Samples: {theta.shape[0]} drawn, {x.shape[0]} simulated
                │     Proposal: {pvec(theta.mean(0))} mean
                │               {pvec(theta.var(0))} var
                │  Simulations: {pvec(x.mean(0))} mean
                │               {pvec(x.var(0))} var
                │          MAP: {pvec(_map)}
            ''')

            theta_redrawn = self.collector.sample(
                proposal, 10_000, # theta.shape[0],
                # show_progress_bars=False,
            )

            # show theta_o-based metrics when available
            if theta_o is not None:
                map_l2_str = 'N/A'

                if _map is not None:
                    map_l2_str = pvec(l2(_map, theta_o))

                printd(f'''
                    │
                    ├─ Calibration loss (theta_o)
                    │
                    │           MAP: {map_l2_str}
                    │  Avg proposal: {pvec(torch.mean(l2(theta_redrawn, theta_o)))}
                ''')


            # show x_o-based metrics when available
            if x_o is not None:
                map_l2_str = 'N/A'

                if _map is not None:
                    # simulate 1/10 of the `num_to_simulate` samples @ MAP
                    x_at_map = self.collector.simulate(
                        _map.repeat((num_to_simulate//10, 1))
                    )

                    map_l2_str = pvec(torch.mean(l2(x_at_map, x_o)))

                avg_post_l2 = torch.mean(l2(x, x_o))

                printd(f'''
                    │
                    ├─ Calibration loss (x_o)
                    │
                    │      xs @ MAP: {map_l2_str}
                    │  Avg proposal: {pvec(avg_post_l2)}
                ''')

                self._seedwise_metrics[seed][i]['l2_xo'] = avg_post_l2

            # print proposal shrinkage if not prior
            if i == 0:
                continue

            printd(f'''
                │
                ├─ Shrinkage
                │
                │     From prior: {pvec(posterior_shrinkage(prior_samples, theta))}
                │  From previous: {pvec(posterior_shrinkage(thetas[i-1], theta))}
            ''')

            # check if SBIBM run & get deeper metrics if so
            if not is_sbibm_run:
                continue

            task = self.si.task
            num_obs = self.si._seedwise_num_observation[seed]
            ref_theta = task.get_reference_posterior_samples(num_observation=num_obs)

            # maybe cut the references?
            # ref_theta = ref_theta[torch.randperm(ref_theta.shape[0])[:theta.shape[0]]]
            ref_theta = ref_theta[torch.randperm(ref_theta.shape[0])[:2500]]
            theta_redrawn = theta_redrawn[torch.randperm(theta_redrawn.shape[0])[:2500]]

            print('Benchmarking c2st...')
            _c2st = c2st(ref_theta, theta_redrawn)[0]
            print('Benchmarking mmd...')
            _mmd = mmd(ref_theta, theta_redrawn)
            print('Benchmarking ksd...')
            _ksd = torch.tensor(ksd(task, num_obs, theta_redrawn))
            print('Benchmarking meddist...')
            _meddist = median_distance(x, x_o)[0]
            print('Benchmarking meanerr...')
            _meanerr = posterior_mean_error(theta_redrawn, ref_theta)

            self._seedwise_metrics[seed][i]['c2st'] = _c2st
            self._seedwise_metrics[seed][i]['mmd'] = _mmd
            self._seedwise_metrics[seed][i]['ksd'] = _ksd
            self._seedwise_metrics[seed][i]['meddist'] = _meddist
            self._seedwise_metrics[seed][i]['meanerr'] = _meanerr

            printd(f'''
                │
                ├─ SBI-BM metrics
                │
                │      C2ST: {pvec(_c2st)}
                │       MMD: {pvec(_mmd)}
                │       KSD: {pvec(_ksd)}
                │   MEDDIST: {pvec(_meddist)}
                │   MEANERR: {pvec(_meanerr)}
            ''')

    def plot_round_proposal(
        self,
        num_samples: int = 10_000,
        from_round: int = -1,
        show_map: bool = False,
        **pairplot_kwargs,
    ):
        '''
        Parameters:
            show_map: plot the MAP, computing it if not available. Uses
                ``num_samples`` as the ``num_init_samples`` for the
                ``.map()`` method, among which we (by default) select the
                100 highest likelihood points to use as starting points
                for the MAP optimization process. NOTE: if also plotting
                other points (e.g., a "true" synthetic parameter used to
                generated x_o), you need to be specific about labels
                passed as ``fig_kwargs`` in order to ensure the MAP plays
                well with your custom points.
        '''
        proposal = self.si.get_round_proposal(from_round)
        proposal_samples = self.sample_proposal(from_round, num_samples)

        if show_map:
            # safely attempt to get map, skipping if prior
            _map = self._get_map(proposal, proposal_samples)
            _map = torch.atleast_2d(_map)

            if _map is not None:
                # update points
                pairplot_kwargs['points'] = [
                    _map
                ] + pairplot_kwargs.get('points', [])

                # update fig_kwargs, respect passed options
                if 'fig_kwargs' not in pairplot_kwargs:
                    pairplot_kwargs['fig_kwargs'] = {}

                fig_kw = pairplot_kwargs['fig_kwargs']
                fig_kw.update({
                    'legend': True,
                    'points_labels': [
                        'MAP'
                    ] + fig_kw.get('points_labels', []),
                    "despine": {"offset": 0},
                })

        self._set_param_labels(pairplot_kwargs)
        fig, ax = pairplot(proposal_samples, **pairplot_kwargs)

    def plot_round_outputs(
        self,
        num_samples: int = 1_000,
        from_round: int = -1,
        **pairplot_kwargs,
    ):
        outputs = self.simulate_proposal(from_round, num_samples)

        self._set_output_labels(pairplot_kwargs)
        fig, ax = pairplot(outputs, **pairplot_kwargs)

    def plot_round_thetas(
        self,
        from_round: int = -1,
        **pairplot_kwargs,
    ):
        round_thetas = self.si.get_round_thetas(from_round)

        self._set_param_labels(pairplot_kwargs)
        fig, ax = pairplot(round_thetas, **pairplot_kwargs)

    def plot_round_xs(
        self,
        from_round: int = -1,
        **pairplot_kwargs,
    ):
        round_xs = self.si.get_round_xs(from_round)

        self._set_output_labels(pairplot_kwargs)
        fig, ax = pairplot(round_xs, **pairplot_kwargs)

    # run-wise plots
    def plot_run_thetas(
        self,
        sample_final_proposal: bool = True,
        num_final_samples: int = 10_000,
        **pairplot_kwargs,
    ):
        '''
        Plot parameter samples drawn across all inference rounds on one plot.
        '''
        # if specific pairplot kwargs not given, use the following defaults
        # (fairly important for comparison and visibility across rounds)
        if 'diag' not in pairplot_kwargs:
            # hist is OK if same number of samples each round
            # but we sample final
            pairplot_kwargs['diag'] = 'kde'

        theta_list = self.si._seedwise_thetas[self.si._seed]

        if sample_final_proposal:
            final_theta = self.collector.sample(
                self.si.get_round_proposal(),
                num_final_samples,
            )
            theta_list = theta_list + [final_theta]

        self._set_param_labels(pairplot_kwargs)
        fig, ax = pairplot(theta_list, **pairplot_kwargs)

    def plot_run_proposals(
        self,
        num_samples: int = 10_000,
        **pairplot_kwargs,
    ):
        '''
        Plot parameter samples drawn across all inference rounds on one plot.
        '''
        num_props = len(self.si._seedwise_proposals[self.si._seed])
        alphas = list(map(lambda x: x/num_props, range(1, num_props+1)))

        # if specific pairplot kwargs not given, use the following defaults
        # (fairly important for comparison and visibility across rounds)
        self._set_multiplot_kwargs(pairplot_kwargs, alphas)
        self._set_param_labels(pairplot_kwargs)

        theta_list = [
            self.sample_proposal(from_round, num_samples)
            for from_round in range(num_props)
        ]

        # reverse color cycle to show later proposals as darker
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS[::-1])
        fig, axes = pairplot(theta_list, **pairplot_kwargs)
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)

    def _set_param_labels(
        self,
        pairplot_kwargs,
    ):
        theta_dim = self.get_theta_dim()

        if 'labels' not in pairplot_kwargs:
            pairplot_kwargs['labels'] = [
                rf"$\theta_{d}$" for d in range(1, theta_dim+1)
            ]

    def _set_output_labels(
        self,
        pairplot_kwargs,
    ):
        x_dim = self.get_x_dim()

        if 'labels' not in pairplot_kwargs:
            pairplot_kwargs['labels'] = [
                rf"$x_{d}$" for d in range(1, x_dim+1)
            ]

    def _set_multiplot_kwargs(
        self,
        pairplot_kwargs,
        alphas: list | None = None,
    ):
        if 'diag' not in pairplot_kwargs:
            pairplot_kwargs['diag'] = 'kde'
        if 'upper' not in pairplot_kwargs:
            pairplot_kwargs['upper'] = 'contour'

        def subplot_kwargs(alpha=1.0):
            return {
                'levels': [0.68],
                'mpl_kwargs': {
                    'alpha': 0.9,
                },
            }

        if alphas is not None:
            if 'diag_kwargs' not in pairplot_kwargs:
                pairplot_kwargs['diag_kwargs'] = [
                    subplot_kwargs(alpha) for alpha in alphas
                ]
            if 'upper_kwargs' not in pairplot_kwargs:
                pairplot_kwargs['upper_kwargs'] = [
                    subplot_kwargs(alpha) for alpha in alphas
                ]
        else:
            if 'diag_kwargs' not in pairplot_kwargs:
                pairplot_kwargs['diag_kwargs'] = subplot_kwargs(0.8)
            if 'upper_kwargs' not in pairplot_kwargs:
                pairplot_kwargs['upper_kwargs'] = subplot_kwargs(0.8)

    def lc2st(
        self,
        from_round: int = -1,
        num_samples: int = 1_000
    ):
        '''
        not complete
        '''
        NUM_TRAIN = 10_000
        NUM_CAL = int(0.1 * NUM_TRAIN) # 10% of the training data
        NUM_EVAL = 10_000

        thetas_star = self.si.prior.sample((3,))
        xs_star = simulator(thetas_star)

        theta_cal = self.sample_proposal(from_round=0, num_samples=num_samples)
        x_cal = self.simulate_proposal(from_round=0, num_samples=num_samples)

        proposal = self.si.get_round_proposal(from_round)
        post_samples_cal = proposal.sample((1,), x_cal).reshape(-1, theta_cal.shape[-1]).detach()

        flow_inverse_transform = lambda theta, x: proposal.net._transform(theta, context=x)[0]
        flow_base_dist = proposal.net._distribution

        # lc2st_nf = LC2ST_NF(
        #     thetas=theta_cal,
        #     xs=x_cal,
        #     posterior_samples=post_samples_cal,
        #     flow_inverse_transform=flow_inverse_transform,
        #     flow_base_dist=flow_base_dist,
        #     num_ensemble=1,
        # )
        # _ = lc2st_nf.train_under_null_hypothesis()
        # _ = lc2st_nf.train_on_observed_data()
        # conf_alpha = 0.05

class BayesianInferenceDiagnostic(InferenceDiagnostic):
    '''
    Extends InferenceDiagnostic to Bayesian posterior estimators.

    Assumes the SequentialInference base being wrapped is a Bayesian model,
    which for now is assumed to be a model with weight samples to be drawn via
    MC dropout.
    '''

    def __init__(
        self,
        si: SequentialInference,
        num_workers: int = 1,
    ):
        self.si = si
        self.collector = BayesianCollector(self.si.simulator, num_workers)

        self._proposal_samples = {}
        self._proposal_simulations = {}

        self._seedwise_metrics = defaultdict(lambda: defaultdict(dict))

    def _create_prior_de(
        self,
        num_prior_samples: int = 8,
    ):
        '''
        Parameters:
            num_prior_samples: samples from the prior and the simulator used to
                infer dimensionality when initializing the NDE. These have no
                other effect on the model, e.g., no training takes place on
                these values.
        '''
        # first collect samples from the prior/simulator
        theta, x = self.collector.collect(self.si.prior, num_prior_samples)

        # explicitly create the DE based on the density builder
        de = self.si._density_builder_wrapper()(theta.to('cpu'), x.to('cpu'))

        # NOTE: we more or less need to replicate sbi's `train()` method, e.g.,
        # that found in `npe_base.py`. This calls the standard equivalent of
        # `_density_builder_wrapper` to create the DE model and ensures
        # everything is created on the CPU before *then* moving it all to the
        # target device. This is actually handled by that `train()` method
        # *outside* of the factory methods like `build_maf`; if it were to
        # handle that, we'd just be able to rely on our `build_bmaf` that is
        # used in the `BayesianSNPE's DE builder method. So this is why we do
        # the explicit CPU move above, and then move the DE back to the SI's
        # target device below.
        de.to(self.si.device)

        # raw, uncorrected
        # x = reshape_to_batch_event(
        #     x, event_shape=de.condition_shape
        # )
        # de.sample((8,), condition=rx).squeeze()

        # construct a posterior using the SI preferences; this should respect
        # parameter prior bounds
        #
        # should have limited effects on the si.inference; not much gets
        # tracked within this method
        # de_prior = self.si.inference.build_posterior(
        de_prior = BayesDirectPosterior(
            posterior_estimator=de,
            prior=self.si.prior,
            device=self.si.device,  # does need explicit device
        )

        return de_prior

    def plot_de_prior_predictive(
        self,
        x_o: Tensor,
        num_weight_samples: int = 100,
        num_dropout_samples: int = 100,
        num_dist_samples: int = 1_000,
        disable_dropout: bool = False,
        plot_limit: int = 10_000,
        de_prior: BayesDirectPosterior | None = None,
        **pairplot_kwargs,
    ):
        '''
        Plot the density estimator's distribution under the MC dropout (weight)
        prior.

        Thus manually constructs a density estimator using the specs available
        in the SI object. This allows us to see the effect of the prior on the
        weights before training or actually running an inference loop.
        Otherwise, the SI object and ``sbi`` inference wrappers don't even
        create the density estimator until we begin a training session.

        NOTE: our prior on network weights p(ϕ) is expressed through the
            randomness in the initialization scheme. While MC dropout plays a
            role in trained models by allowing for approximate draws of p(ϕ|D),
            the situation is different prior to training. Here we can directly
            sample from p(ϕ), and while applying dropout over realized models
            helps demonstrate some kind of structural uncertainty, it isn't
            strictly a necessary part of capturing the prior predictive
            distribution. In our case, we're interested in the kinds of prior
            models possible p(θ|x,ϕ), ϕ ~ p(ϕ), giving us a sense of the
            prior predictive p(θ|x).

            That being said, using MC dropout in the prior case allows us to
            express a structural uncertainty over the model topology
            explicitly. Provided we are training with dropout (and in general
            may find that some nodes aren't needed to fully express the desired
            relationship), applying random dropout masks over a fixed network 
            ϕ ~ p(ϕ) allows us to cut up the network in several plausible ways,
            realizing a kind of uncertainty we have about the final network
            structure. Our dropout probability is thus capturing some prior
            uncertainty about how likely any particular node is to contribute
            to the final functional model, and when this is > 0, we're saying
            we aren't too sure about what the final network topology should
            look like.

            Thus, we can get a sense for the prior predictive by incorporating
            *both* a weight uncertainty (captured by the initialization scheme
            p(ϕ)) and a structural/topological uncertainty (captured by
            applying random dropout masks with a particular probability; let
            p(z) be the probability of drawing a particular *mask*, which is
            typically factored node-wise by just setting a single Bernoulli
            probability p). Our final representation of the prior predictive is
            then

            p(θ∣x)=∬p(θ|x,ϕ,z)p(z)p(ϕ)dϕdz

            We approximate this by randomly initializing several models and
            applying several random dropout masks on each. Each model gives us
            a *full* p(θ|x,ϕ,z) because we're using a flow, which we then
            sample.

        NOTE: Although we manually create a density estimator, we still attempt
        to wrap it using the inference wrapper from the parent SI object. This
        allows us to see the effect of the prior on the DE weights in a way
        that respects the parameter prior p(θ). That is, we draw ϕ ~ p(ϕ) and
        use an arbitrary x to produce θ ~ p(θ|x,ϕ). When unconstrained, these θ
        may lay outside the support of p(θ), so any leakage under models coming
        from p(ϕ) are corrected for when we plot this.

        Parameters:
            num_weight_samples: the number of draws from p(ϕ), i.e., how many times we
                re-initialize our model.
            num_dropout_samples: the number of random dropout masks to apply
                under each initialized model
            num_dist_samples: the number of samples to draw from the produced
                distribution. This is the number of params sampled from
                p(θ|x,ϕ,z) if ``disable_dropout=False``, or p(θ|x,ϕ) if
                ``disable_dropout=True``.
            disable_dropout: whether to use dropout in order to incorporate a
                "structural uncertainty," i.e., producing an estimate of p(θ∣x)
                using p(θ|x,ϕ,z) rather than just p(θ|x,ϕ)
            plot_limit: the max number of samples to use when generating plots
                (prevents huge possible wait times when # weight and # dropout
                samples is prohibitively large altogether)
            de_prior: reference to an existing DE (of type
                BayesDirectPosterior) to use when drawing samples
        '''
        sample_list = []
        log_probs_list = []
        for _ in tqdm(range(num_weight_samples)):
            # create wrapped DE prior
            if de_prior is None:
                de_prior = self._create_prior_de()

            if disable_dropout:
                de_prior.posterior_estimator.eval()
            else:
                # ensure dropout is active
                de_prior.posterior_estimator.train()


            # activations = {}
            # model = de_prior.posterior_estimator

            # def hook_fn(module, input, output):
            #     activations['dropout_output'] = output

            # for i, block in enumerate(model.net._transform._transforms[1].autoregressive_net.blocks):
            #     hook = block.mc_dropout.register_forward_hook(hook_fn)


            with torch.no_grad():
                # when dropout active, we batch of samples is drawn from the
                # distribution under a particular dropout mask, p(θ|x,ϕ,z)

                #dropout_samples = de_prior.sample_batched(
                dropout_samples, log_probs = de_prior.sample_and_log_prob_batched(
                    (num_dist_samples,),
                    x=x_o.repeat(num_dropout_samples, 1),

                    # this destroys isolated dropout masks if too small
                    # max_sampling_batch_size=1_000,  # more efficient, apparently

                    show_progress_bars=False,
                )  # with dim N x Z x |θ| (odd sbi ordering)

                # fix with transpose; producing dim Z x N x |θ|
                dropout_samples = dropout_samples.transpose(0, 1)
                log_probs = log_probs.transpose(0, 1)

                # print(
                #    (activations['dropout_output'] == 0) .sum(dim=1).unique()
                # )

                # dropout_samples = []
                # for _ in range(num_dropout_samples):
                #     dropout_samples.append(
                #         de_prior.sample(
                #             (num_dist_samples,),
                #             x=x_o,
                #             show_progress_bars=False,
                #         )
                #     )
                #     #print(
                #     #    (activations['dropout_output'] == 0) .sum(dim=1).unique()
                #     #)

                # dropout_samples = torch.stack(dropout_samples)

            sample_list.append(dropout_samples)
            log_probs_list.append(log_probs)

        # final dim W x Z x N x |θ| 
        samples = torch.stack(sample_list)
        log_probs = torch.stack(log_probs_list)
        print(f'Total sample dimensions (W, Z, N, |θ|) : {list(samples.shape)}')

        self._set_multiplot_kwargs(pairplot_kwargs)

        # plot total predictive prior; flatten to (WxZxN) x |θ| 
        pp_samples = torch.flatten(samples, end_dim=-2)
        pp_samples = pp_samples[torch.randperm(pp_samples.size(0))]
        pp_samples = pp_samples[:plot_limit]
        print(f'Plot 1: p(θ|x) with shape {list(pp_samples.shape)}')

        fig, ax = pairplot(
            pp_samples,
            **{
                **pairplot_kwargs,
                'upper_kwargs': {'levels': [0.1, 0.68]},
            }
        )

        # plot some specific weight draws, marginalizing the structural samples
        # flatten to W[:3] x (ZxN) x |θ| 
        w_samples = torch.flatten(samples[:3], start_dim=1, end_dim=2)
        w_samples = w_samples[:,:plot_limit,:]
        print(f'Plot 2: p(θ|x,ϕ) for 3 ϕ~p(ϕ), shape {list(w_samples.shape)}')

        fig, ax = pairplot(
            list(w_samples),
            **pairplot_kwargs,
        )

        # for a few weight draws, plot a few specific structural dist.s
        # flatten to W[:3] x Z[:3] x N x |θ| 

        wz_samples = torch.flatten(samples[:3, :3, :plot_limit], start_dim=0, end_dim=1)

        # wz_samples = torch.flatten(samples.permute(0,2,1,3)[:3,:3], start_dim=0, end_dim=1)
        # wz_samples = wz_samples[:,:plot_limit,:]

        print(f'Plot 3: p(θ|x,ϕ,z) for 3 z~p(z), 3 ϕ~p(ϕ), shape {list(wz_samples.shape)}')

        uk = pairplot_kwargs['upper_kwargs']
        pairplot_kwargs['upper_kwargs'] = [
            {**uk, "mpl_kwargs": {'alpha':0.5, 'cmap': None, 'colors': COLORS[0+5*0]}},
            {**uk, "mpl_kwargs": {'alpha':0.5, 'cmap': None, 'colors': COLORS[0+5*0]}},
            {**uk, "mpl_kwargs": {'alpha':0.5, 'cmap': None, 'colors': COLORS[0+5*0]}},

            {**uk, "mpl_kwargs": {'alpha':0.5, 'cmap': None, 'colors': COLORS[2+5*0]}},
            {**uk, "mpl_kwargs": {'alpha':0.5, 'cmap': None, 'colors': COLORS[2+5*0]}},
            {**uk, "mpl_kwargs": {'alpha':0.5, 'cmap': None, 'colors': COLORS[2+5*0]}},

            {**uk, "mpl_kwargs": {'alpha':0.5, 'cmap': None, 'colors': COLORS[4+5*0]}},
            {**uk, "mpl_kwargs": {'alpha':0.5, 'cmap': None, 'colors': COLORS[4+5*0]}},
            {**uk, "mpl_kwargs": {'alpha':0.5, 'cmap': None, 'colors': COLORS[4+5*0]}},
        ]

        fig, ax = pairplot(
            list(wz_samples),
            **pairplot_kwargs
        )

        # log_probs = None
        return samples, log_probs

    def plot_round_proposal_components(
        self,
        num_samples: int = 1_000,
        from_round: int = -1,
        num_components: int = 3,
        overlay_pp: bool = False,
        **pairplot_kwargs,
    ):
        '''
        Plots several individual proposal components.

        Here we draw batches of samples for ``num_component`` individual
        samples from the weight posterior. This allows us to get a sense for
        specific weight realizations and how much they appear to vary for
        different draws ϕ ~ p(ϕ|D).

        If ``overlay_pp`` is set, we also plot the full posterior predictive
        p(θ|x,D) atop the individual predictive distributions p(θ|x,ϕ) that
        it comprises. The full posterior predictive is sampled from the general
        ``BayesianCollector.sample(...)`` and thus isn't only collected from
        the few specific predictive distributions we're showing alongside it
        (that is, we compute the full PP like usual, not just from the
        components).

        NOTE: this makes use of ``sbi.pairplot's`` multi-sample functionality.
        Each of the requested groups of points are bundled up as a list, and
        passed as a whole to the ``pairplot`` call to be overlain on the same
        plot. The provided ``pairplot_kwargs`` thus should reflect the styling
        desired for each of these groups: for most options, you can pass a list
        of styles to be apply to the respective group (e.g., if you're plotting
        3 components, you could pass ``upper=['kde','hist','kde']`` to use
        specify the display method to use for each). There should be color
        variability by default.
        '''
        proposal = self.si.get_round_proposal(from_round)

        samples = [
            self.collector.sample_component(proposal, num_samples)
            for _ in range(num_components)
        ]

        if overlay_pp:
            samples.append(
                self.sample_proposal(from_round, num_samples)
            )

        self._set_multiplot_kwargs(pairplot_kwargs)
        self._set_param_labels(pairplot_kwargs)

        fig, ax = pairplot(samples, **pairplot_kwargs)

    def plot_round_proposal_component_outputs(
        self,
        num_samples: int = 100,
        from_round: int = -1,
        num_components: int = 3,
        overlay_pp: bool = False,
        **pairplot_kwargs,
    ):
        proposal = self.si.get_round_proposal(from_round)

        samples = [
            self.collector.collect_component(proposal, num_samples)[1]
            for _ in range(num_components)
        ]

        if overlay_pp:
            samples.append(
                self.simulate_proposal(from_round, num_samples)
            )

        self._set_multiplot_kwargs(pairplot_kwargs)
        self._set_output_labels(pairplot_kwargs)

        fig, ax = pairplot(samples, **pairplot_kwargs)

    def pp_entropy(
        self,
        from_round=-1,
        num_samples=10_000,
    ):
        proposal = self.si.get_round_proposal(from_round)
        x_o = proposal.default_x
        proposal.posterior_estimator.train()

        with torch.no_grad():
            # of size (K, |x_o|)
            batched_x = x_o.repeat(num_samples, 1)

            # θ ~ p(θ|x_o,D) via θ ~ p(θ|x_o,ϕ) <- ϕ ~ p(ϕ|D)
            batched_theta = proposal.sample_batched(
                (1,),
                x=batched_x,
                max_sampling_batch_size=100,  # more efficient, apparently
                show_progress_bars=False
            )  # produces size (1, K, |θ|),

            # get size (K, K, |θ|)
            batched_theta = batched_theta.repeat(num_samples, 1, 1)

            log_probs = proposal.log_prob_batched(
                batched_theta,
                batched_x,
            )  # produces size (1, K)

            comp_entropies = -log_probs.sum(dim=1)

        return comp_entropies

    def component_entropies(
        self,
        from_round=-1,
        num_components=1_000,
        num_component_samples=1_000,
    ):
        '''
        Comment notation:

        - K: # of individual draws ϕ ~ p(ϕ|D)
        - M: # of samples drawn from each component, e.g., θ ~ p(θ|x,ϕ[i])

        So we draw K models, and samples M parameters from each of those
        models. We then evaluate the log-probs 
        '''
        proposal = self.si.get_round_proposal(from_round)
        x_o = proposal.default_x
        proposal.posterior_estimator.train()

        with torch.no_grad():
            # of size (K, |x_o|)
            batched_x = x_o.repeat(num_components, 1)

            # θ ~ p(θ|x_o,D) via θ ~ p(θ|x_o,ϕ) <- ϕ ~ p(ϕ|D)
            batched_theta = proposal.sample_batched(
                (num_component_samples,),
                x=batched_x,
                max_sampling_batch_size=100,  # more efficient, apparently
                show_progress_bars=False
            )  # produces size (M, K, |θ|),

            log_probs = proposal.log_prob_batched(
                batched_theta,
                batched_x,
            )  # produces size (M, K)

            comp_entropies = -log_probs.sum(dim=1)

        return comp_entropies
