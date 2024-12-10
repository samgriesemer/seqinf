import time
import logging

import numpy as np

import torch
from tqdm.auto import tqdm

from sbi.utils.sbiutils import seed_all_backends
from sbi.inference.posteriors.base_posterior import NeuralPosterior

from seqinf.util import simulate_in_batches


logger = logging.getLogger(__name__)


class Collector:
    '''
    Diff between sim_in_batches and sim_for_sbi?

    check execlib here, should work as threaded if sim wraps a subproc call
    '''

    def __init__(
        self,
        simulator,
        num_workers: int = 1,
        show_progress_bar: bool = True,
    ):
        self.simulator = simulator
        self.num_workers = num_workers
        self.show_progress_bar = show_progress_bar

    def _is_prior(self, proposal) -> bool:
        '''
        Check if the proposal appears to be a prior.

        This is needed for the ensemble case, where we want ``.sample()`` to
        also work for the prior which is a concrete torch.Distribution.
        '''
        return not isinstance(proposal, NeuralPosterior)

    def sample(
        self,
        proposal,
        num_samples: int,
        seed: int | None = None,
        show_progress_bars: bool | None = True
    ):
        seed_all_backends(seed)

        # require the proposal to be "pre-conditioned" with our x_o
        if not self._is_prior(proposal) and proposal.default_x is None:
            raise ValueError(
                'Non-prior proposal needs to have x_o set; '
                'use `set_default_x` first'
            )

        return proposal.sample(
            (num_samples,),
            # show_progress_bars=show_progress_bars
        )

    def simulate(
        self,
        theta,
        sim_batch_size: int | None = -1,
        seed:           int | None = None,
    ):
        return simulate_in_batches(
            self.simulator,
            theta,
            num_workers=self.num_workers,
            simulation_batch_size=sim_batch_size,
            seed=seed,
            show_progress_bar=self.show_progress_bar,
        )

    def collect(
        self,
        proposal,
        num_samples,
        sim_batch_size: int | None = -1,
        seed:           int | None = None,
    ):
        '''
        Draw parameters from the proposal and run the simulator.

        Parameters:
            proposal:
                proposal distribution to sample parameters from
            num_samples:
                number of samples to draw from the proposal
            sim_batch_size:
                size of the param batches to use when running the simulator. By
                default set to ``-1``, which will attempt to evenly split the
                ``num_samples`` into ``num_workers`` total batches.
            seed:
                random seed to use when sampling and simulating
        '''
        # both calls here are seeded, dont need a separate backend seed call
        theta = self.sample(proposal, num_samples, seed)
        x = self.simulate(theta, sim_batch_size, seed)

        return theta, x


class BayesianCollector(Collector):
    def sample_component(
        self,
        proposal,
        num_samples: int,
        seed:        int | None = None,
    ):
        '''
        Draws samples from a single component of the ensemble.

        When using MC dropout, this lets us sample from a distribution that
        results from a fixed weight draw.
        '''
        seed_all_backends(seed)

        # keep dropout active
        proposal.posterior_estimator.train()

        with torch.no_grad():
            # draw a single batch under same weights
            samples = proposal.sample((num_samples,))

        proposal.posterior_estimator.eval()

        return samples

    def collect_component(
        self,
        proposal,
        num_samples,
        sim_batch_size: int | None = -1,
        seed:           int | None = None,
    ):
        '''
        Draw parameters from a proposal component and run the simulator.

        Allows us to simulate parameters that originate from an individual
        predictive distribution p(θ|x,ϕ). We need this dedicated method, as the
        parent ``.collect()`` draws samples from the overloaded ``.sample()``,
        which draw parameters from p(θ|x,D).

        Parameters:
            proposal:
                proposal distribution to sample parameters from
            num_samples:
                number of samples to draw from the proposal
            sim_batch_size:
                size of the param batches to use when running the simulator. By
                default set to ``-1``, which will attempt to evenly split the
                ``num_samples`` into ``num_workers`` total batches.
            seed:
                random seed to use when sampling and simulating
        '''
        # both calls here are seeded, dont need a separate backend seed call
        theta = self.sample_component(proposal, num_samples, seed)

        x = simulate_in_batches(
            self.simulator,
            theta,
            num_workers=self.num_workers,
            simulation_batch_size=sim_batch_size,
            seed=seed,
            show_progress_bar=self.show_progress_bar,
        )

        return theta, x

    def sample(
        self,
        proposal,
        num_samples: int,
        seed:        int | None = None,
    ):
        '''
        Sample from an ensemble/Bayesian model via MC dropout.

        Collection takes place by aggregating samples like

        θ ~ p(θ|x,ϕ), ϕ ~ p(ϕ|D)

        This sampling takes place for every sample drawn; we do not fix the
        weights (see ``sample_component()`` to do this). Keeping dropout active
        means each model query is (approximately) the result of a different
        draw under the weight posterior, and we take a single parameter sample
        from the resulting distribution.
        '''
        seed_all_backends(seed)

        if self._is_prior(proposal):
            return super().sample(proposal, num_samples, seed)

        # require the proposal to be "pre-conditioned" with our x_o
        if proposal.default_x is None:
            raise ValueError(
                'Non-prior proposal needs to have x_o set; '
                'use `set_default_x` first'
            )

        x_o = proposal.default_x

        # keep dropout active
        proposal.posterior_estimator.train()

        # spread x_o out across num_samples and sample in batches
        # each θ|x_o will be sampled using a different ϕ ~ p(ϕ|D)
        samples = []
        with torch.no_grad():
            # tqdm_desc = f'Drawing {num_samples} proposal samples'
            # for _ in tqdm(range(num_samples), desc=tqdm_desc):
            #     samples.append(
            #         proposal.sample(
            #             (1,),
            #             show_progress_bars=False
            #         )
            #     )

            # this is very slow for some reason? TODO: revisit
            logger.info(f'Sampling {num_samples} batched p(θ|x_o,D) params')
            samples = proposal.sample_batched(
                (1,),
                x=x_o.repeat(num_samples, 1),
                max_sampling_batch_size=100,
                show_progress_bars=False
            ).squeeze()  # remove the lead dim in 1xNx|θ|

        # proposal.posterior_estimator.eval()

        # return torch.vstack(samples)
        return samples

    def sample_select(
        self,
        proposal,
        num_samples: int,
        num_components: int,
    ):
        pass


class ASNPECollector(BayesianCollector):
    def sample(
        self,
        proposal,
        num_samples,
        seed: int | None = None,
    ):
        '''
        Proposal is presumably a conditioned DirectPosterior, which will correct log-probs
        and sampling for leakage in the underlying density estimator (happening with
        bounded priors).
        '''
        seed_all_backends(seed)

        if self._is_prior(proposal):
            return super().sample(proposal, num_samples, seed)

        # require the proposal to be "pre-conditioned" with our x_o
        if proposal.default_x is None:
            raise ValueError(
                'Non-prior proposal needs to have x_o set; '
                'use `set_default_x` first'
            )

        x_o = proposal.default_x

        pool_mult = 8
        frozen_frac = 0.85
        pool_size = num_samples * pool_mult
        samples = super().sample(proposal, pool_size, seed)

        num_frozen = int(num_samples*frozen_frac)
        num_to_select = num_samples - num_frozen

        # split pool
        frozen = samples[:num_frozen]
        candidates = samples[num_frozen:]

        proposal.posterior_estimator.train()
        # mcd_seed = np.random.randint(2**30)

        # N: int = candidates.shape[0]
        # M: int = 256
        K: int = 128

        start = time.time()
        cand_K = candidates.unsqueeze(1).repeat((1, K, 1))
        log_prob_NK = proposal.log_prob_batched(
            cand_K,
            x=x_o.repeat((K, 1)),
            norm_posterior=False,  # True,
            # mcd_seed=mcd_seed
        )
        log_prob_vars = log_prob_NK.var(dim=1)
        # log_prob_vars = log_prob_NK.exp().mean(dim=1)*log_prob_NK.exp().var(dim=1)
        print(f'Log-prob NxK took {time.time()-start}')

        top_candidates = candidates[log_prob_vars.topk(num_to_select).indices]

        return torch.vstack((frozen, top_candidates))
