import logging
from typing import Callable

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import Tensor, as_tensor, nn, log
from torch.distributions import Distribution

from sbi.sbi_types import Shape
from sbi.inference import DirectPosterior
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.utils.sbiutils import within_support
from sbi.utils.torchutils import ensure_theta_batched


class BayesDirectPosterior(DirectPosterior):

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Tensor | None = None,
        max_sampling_batch_size: int = 10_000,
        sample_with: str | None = None,
        show_progress_bars: bool = True,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ) -> Tensor:
        r"""Return samples from posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            sample_with: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. If it is set, we instantly raise an error.
            show_progress_bars: Whether to show sampling progress monitor.
        """
        num_samples = torch.Size(sample_shape).numel()
        x = self._x_else_default_x(x)
        x = reshape_to_batch_event(
            x, event_shape=self.posterior_estimator.condition_shape
        )
        if x.shape[0] > 1:
            raise ValueError(
                ".sample() supports only `batchsize == 1`. If you intend "
                "to sample multiple observations, use `.sample_batched()`. "
                "If you intend to sample i.i.d. observations, set up the "
                "posterior density estimator with an appropriate permutation "
                "invariant embedding net."
            )

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported. You have to rerun "
                f"`.build_posterior(sample_with={sample_with}).`"
            )

        samples = accept_reject_sample(
            proposal=self.posterior_estimator,
            accept_reject_fn=lambda theta: within_support(self.prior, theta),
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            max_sampling_batch_size=max_sampling_batch_size,
            proposal_sampling_kwargs={"condition": x},
            alternative_method="build_posterior(..., sample_with='mcmc')",
            # mcd_seed=mcd_seed,
        )[0]  # [0] to return only samples, not acceptance probabilities.

        return samples[:, 0]  # Remove batch dimension.

    def log_prob(
        self,
        theta: Tensor,
        x: Tensor | None = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: dict | None = None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ) -> Tensor:
        r"""Returns the log-probability of the posterior $p(\theta|x)$.

        Args:
            theta: Parameters $\theta$.
            norm_posterior: Whether to enforce a normalized posterior density.
                Renormalization of the posterior is useful when some
                probability falls out or leaks out of the prescribed prior support.
                The normalizing factor is calculated via rejection sampling, so if you
                need speedier but unnormalized log posterior estimates set here
                `norm_posterior=False`. The returned log posterior is set to
                -∞ outside of the prior support regardless of this setting.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            leakage_correction_params: A `dict` of keyword arguments to override the
                default values of `leakage_correction()`. Possible options are:
                `num_rejection_samples`, `force_update`, `show_progress_bars`, and
                `rejection_sampling_batch_size`.
                These parameters only have an effect if `norm_posterior=True`.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """
        x = self._x_else_default_x(x)

        theta = ensure_theta_batched(torch.as_tensor(theta))
        theta_density_estimator = reshape_to_sample_batch_event(
            theta, theta.shape[1:], leading_is_sample=True
        )
        x_density_estimator = reshape_to_batch_event(
            x, event_shape=self.posterior_estimator.condition_shape
        )
        if x_density_estimator.shape[0] > 1:
            raise ValueError(
                ".log_prob() supports only `batchsize == 1`. If you intend "
                "to evaluate given multiple observations, use `.log_prob_batched()`. "
                "If you intend to evaluate given i.i.d. observations, set up the "
                "posterior density estimator with an appropriate permutation "
                "invariant embedding net."
            )

        # self.posterior_estimator.eval()

        # fix a constant MC Dropout mask seed, so the same "ensembled" is used
        # for both the log-prob evaluation and samples drawn for leakage
        # correction
        if mcd_seed is None:
            mcd_seed = np.random.randint(2**30)

        with torch.set_grad_enabled(track_gradients):
            # Evaluate on device, move back to cpu for comparison with prior.
            unnorm_log_prob = self.posterior_estimator.log_prob(
                theta_density_estimator,
                condition=x_density_estimator,
                mcd_num_batches=mcd_num_batches,
                mcd_seed=mcd_seed,
            )
            # `log_prob` supports only a single observation (i.e. `batchsize==1`).
            # We now remove this additional dimension.
            unnorm_log_prob = unnorm_log_prob.squeeze(dim=1)

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            masked_log_prob = torch.where(
                in_prior_support,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )

            if leakage_correction_params is None:
                leakage_correction_params = dict()  # use defaults

            log_factor = (
                log(
                    self.leakage_correction(
                        x=x,
                        mcd_num_batches=mcd_num_batches,
                        mcd_seed=mcd_seed,
                        **leakage_correction_params
                    )
                ) if norm_posterior
                else 0
            )

            return masked_log_prob - log_factor

    def sample_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        max_sampling_batch_size: int = 10_000,
        show_progress_bars: bool = True,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ) -> Tensor:
        r"""Given a batch of observations [x_1, ..., x_B] this function samples from
        posteriors $p(\theta|x_1)$, ... ,$p(\theta|x_B)$, in a batched (i.e. vectorized)
        manner.

        Args:
            sample_shape: Desired shape of samples that are drawn from the posterior
                given every observation.
            x: A batch of observations, of shape `(batch_dim, event_shape_x)`.
                `batch_dim` corresponds to the number of observations to be drawn.
            max_sampling_batch_size: Maximum batch size for rejection sampling.
            show_progress_bars: Whether to show sampling progress monitor.

        Returns:
            Samples from the posteriors of shape (*sample_shape, B, *input_shape)
        """
        num_samples = torch.Size(sample_shape).numel()
        condition_shape = self.posterior_estimator.condition_shape
        x = reshape_to_batch_event(x, event_shape=condition_shape)

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        samples = accept_reject_sample(
            proposal=self.posterior_estimator,
            accept_reject_fn=lambda theta: within_support(self.prior, theta),
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            max_sampling_batch_size=max_sampling_batch_size,
            proposal_sampling_kwargs={"condition": x},
            alternative_method="build_posterior(..., sample_with='mcmc')",
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )[0]

        return samples

    def log_prob_batched(
        self,
        theta: Tensor,
        x: Tensor,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: dict | None = None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ) -> Tensor:
        """
        CUSTOM NOTE: Need also to redefine batched log-prob, as we must pass
        batch indices through to the ``BayesFlow.log_prob`` method.

        Given a batch of observations [x_1, ..., x_B] and a batch of parameters \
        [$\theta_1$,..., $\theta_B$] this function evalautes the log-probabilities \
        of the posteriors $p(\theta_1|x_1)$, ..., $p(\theta_B|x_B)$ in a batched \
        (i.e. vectorized) manner.

        Args:
            theta: Batch of parameters $\theta$ of shape \
                `(*sample_shape, batch_dim, *theta_shape)`.
            x: Batch of observations $x$ of shape \
                `(batch_dim, *condition_shape)`.
            norm_posterior: Whether to enforce a normalized posterior density.
                Renormalization of the posterior is useful when some
                probability falls out or leaks out of the prescribed prior support.
                The normalizing factor is calculated via rejection sampling, so if you
                need speedier but unnormalized log posterior estimates set here
                `norm_posterior=False`. The returned log posterior is set to
                -∞ outside of the prior support regardless of this setting.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            leakage_correction_params: A `dict` of keyword arguments to override the
                default values of `leakage_correction()`. Possible options are:
                `num_rejection_samples`, `force_update`, `show_progress_bars`, and
                `rejection_sampling_batch_size`.
                These parameters only have an effect if `norm_posterior=True`.

        Returns:
            `(len(θ), B)`-shaped log posterior probability $\\log p(\theta|x)$\\ for θ \
            in the support of the prior, -∞ (corresponding to 0 probability) outside.
        """

        theta = ensure_theta_batched(torch.as_tensor(theta))
        event_shape = self.posterior_estimator.input_shape
        theta_density_estimator = reshape_to_sample_batch_event(
            theta, event_shape, leading_is_sample=True
        )
        x_density_estimator = reshape_to_batch_event(
            x, event_shape=self.posterior_estimator.condition_shape
        )

        # self.posterior_estimator.eval()

        # fix a constant MC Dropout mask seed, so the same "ensembled" is used
        # for both the log-prob evaluation and samples drawn for leakage
        # correction
        if mcd_seed is None:
            mcd_seed = np.random.randint(2**30)

        with torch.set_grad_enabled(track_gradients):
            # Evaluate on device, move back to cpu for comparison with prior.
            unnorm_log_prob = self.posterior_estimator.log_prob(
                theta_density_estimator,
                condition=x_density_estimator,
                mcd_num_batches=mcd_num_batches,
                mcd_seed=mcd_seed,
            )

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            masked_log_prob = torch.where(
                in_prior_support,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )

            if leakage_correction_params is None:
                leakage_correction_params = dict()  # use defaults
                
            log_factor = (
                log(
                    self.leakage_correction(
                        x=x,
                        mcd_num_batches=mcd_num_batches,
                        mcd_seed=mcd_seed,
                        **leakage_correction_params
                    )
                ) if norm_posterior
                else 0
            )

            return masked_log_prob - log_factor

    def sample_and_log_prob_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        max_sampling_batch_size: int = 10_000,
        show_progress_bars: bool = True,
        norm_posterior: bool = True,
        leakage_correction_params: dict | None = None,
        track_gradients: bool = False,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Given a batch of observations [x_1, ..., x_B], this function samples from
        posteriors $p(\theta|x_1)$, ..., $p(\theta|x_B)$, and returns both samples and their
        log-probabilities in a batched (i.e., vectorized) manner.

        Internal note: does not explicitly enforce `.eval()` to allow active
            dropout.

        Args:
            sample_shape: Desired shape of samples that are drawn from the posterior
                given every observation.
            x: A batch of observations, of shape `(batch_dim, *condition_shape)`.
                `batch_dim` corresponds to the number of observations.
            max_sampling_batch_size: Maximum batch size for sampling (unused here but kept for consistency).
            show_progress_bars: Whether to show sampling progress monitor (unused here but kept for consistency).
            norm_posterior: Whether to enforce a normalized posterior density.
                Renormalization of the posterior is useful when some probability leaks out of
                the prescribed prior support. The normalizing factor is calculated via
                rejection sampling, so if you need speedier but unnormalized log posterior
                estimates, set `norm_posterior=False`. The returned log posterior is set to
                -∞ outside of the prior support regardless of this setting.
            leakage_correction_params: A `dict` of keyword arguments to override the
                default values of `leakage_correction()`. Possible options are:
                `num_rejection_samples`, `force_update`, `show_progress_bars`, and
                `rejection_sampling_batch_size`. These parameters only have an effect if
                `norm_posterior=True`.
            track_gradients: Whether the returned tensors support tracking gradients.
                This can be helpful for e.g., sensitivity analysis, but increases memory
                consumption.

        Returns:
            samples: Samples from the posteriors of shape (*sample_shape, batch_dim, *theta_shape)
            log_probs: Corresponding log probabilities of shape (*sample_shape, batch_dim)
        """
        num_samples = torch.Size(sample_shape).numel()
        condition_shape = self.posterior_estimator.condition_shape
        x = reshape_to_batch_event(x, event_shape=condition_shape)

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        # NOTE: `acceptance_rates` is a (B,) sized tensor of acceptance rates
        # batch-by-batch, and is used to normalize each component. Note that,
        # even if we have fewer components than `x`'s batch size (i.e., we use
        # the same component for several condition rows), we still normalize
        # all separately (which would indeed be required if the rows are
        # different x_o, but we *can* have the same x_o rows being handled by
        # the same component and thus use different normalization constants for
        # the same p(θ|x,ϕ), which should be the same but in practice won't
        # vary much and shouldn't be too problematic)
        samples, log_probs, acceptance_rates = accept_reject_sample_and_log_prob(
            proposal=self.posterior_estimator,
            condition=x,
            accept_reject_fn=lambda theta: within_support(self.prior, theta),
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            max_sampling_batch_size=max_sampling_batch_size,
            alternative_method="build_posterior(..., sample_with='mcmc')",
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )

        log_factor = acceptance_rates.log()

        return samples, (log_probs - log_factor)

    @torch.no_grad()
    def leakage_correction(
        self,
        x: Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        show_progress_bars: bool = False,
        rejection_sampling_batch_size: int = 10_000,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ) -> Tensor:
        r"""Return leakage correction factor for a leaky posterior density estimate.

        The factor is estimated from the acceptance probability during rejection
        sampling from the posterior.

        This is to avoid re-estimating the acceptance probability from scratch
        whenever `log_prob` is called and `norm_posterior=True`. Here, it
        is estimated only once for `self.default_x` and saved for later. We
        re-evaluate only whenever a new `x` is passed.

        Arguments:
            num_rejection_samples: Number of samples used to estimate correction factor.
            show_progress_bars: Whether to show a progress bar during sampling.
            rejection_sampling_batch_size: Batch size for rejection sampling.

        Returns:
            Saved or newly-estimated correction factor (as a scalar `Tensor`).
        """

        def acceptance_at(x: Tensor) -> Tensor:
            # [1:] to remove batch-dimension for `reshape_to_batch_event`.
            return accept_reject_sample(
                proposal=self.posterior_estimator,
                accept_reject_fn=lambda theta: within_support(self.prior, theta),
                num_samples=num_rejection_samples,
                show_progress_bars=show_progress_bars,
                sample_for_correction_factor=True,
                max_sampling_batch_size=rejection_sampling_batch_size,
                proposal_sampling_kwargs={
                    "condition": reshape_to_batch_event(
                        x, event_shape=self.posterior_estimator.condition_shape
                    )
                },
                mcd_num_batches=mcd_num_batches,
                mcd_seed=mcd_seed,
            )[1]

        # Check if the provided x matches the default x (short-circuit on identity).
        is_new_x = self.default_x is None or (
            x is not self.default_x 
            # remove below condition for sake of batches
            # and (x != self.default_x).any() 
        )

        not_saved_at_default_x = self._leakage_density_correction_factor is None

        if is_new_x:  # Calculate at x; don't save.
            return acceptance_at(x)
        elif not_saved_at_default_x or force_update:  # Calculate at default_x; save.
            assert self.default_x is not None
            self._leakage_density_correction_factor = acceptance_at(self.default_x)

        return self._leakage_density_correction_factor  # type: ignore


@torch.no_grad()
def accept_reject_sample(
    proposal: nn.Module | Distribution,
    accept_reject_fn: Callable,
    num_samples: int,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    sample_for_correction_factor: bool = False,
    max_sampling_batch_size: int = 10_000,
    proposal_sampling_kwargs: dict | None = None,
    alternative_method: str | None = None,
    mcd_num_batches: int | None = None,
    mcd_seed: int | None = None,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    r"""
    Returns samples from a proposal according to an acceptance criterion.

    BEGIN INTERNAL NOTE

    The original method, found in ``sbi.rejection``, is only
    ever *fully* called by the ``RejectionPosterior``. Every other case only
    cares about the samples, not the acceptance rate. We modify this
    implementation to return acceptance rates for each batch.

    Additionally, when sampling by batch, we modify the original method such
    that we only track the total remaining samples needed for the smallest
    batch size, rather than using the minimum number of samples accepted at any
    given round. Without this, the sample collection loop can run forever when
    ``num_samples`` is small enough and the ``num_xos`` is large enough: you
    are almost certain to have *some* condition/component have 0 accepted
    samples, and the loop will perceive this as 0 new samples collected. In
    reality, you can have some condition/component with 0 accepted samples
    every time, and still build up to the requested ``num_samples`` quota
    across several iterations (so long as the batch with 0 new samples isn't
    the same each round).

    END INTERNAL NOTE

    This is relevant for snpe methods and flows for which the posterior tends to have
    mass outside the prior support.

    This function could in principle be integrated into `rejection_sample()`. However,
    to keep the warnings clean, to avoid additional code for integration, and confusing
    if-cases, we decided to keep two separate functions.

    This function uses rejection sampling with samples from posterior in order to
        1) obtain posterior samples within the prior support, and
        2) calculate the fraction of accepted samples as a proxy for correcting the
           density during evaluation of the posterior.

    Args:
        posterior_nn: Neural net representing the posterior.
        accept_reject_fn: Function that evaluatuates which samples are accepted or
            rejected. Must take a batch of parameters and return a boolean tensor which
            indicates which parameters get accepted.
        num_samples: Desired number of samples.
        show_progress_bars: Whether to show a progressbar during sampling.
        warn_acceptance: A minimum acceptance rate under which to warn about slowness.
        sample_for_correction_factor: True if this function was called by
            `leakage_correction()`. False otherwise. Will be used to adapt the leakage
             warning and to decide whether we have to search for the maximum.
        max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.
        proposal_sampling_kwargs: Arguments that are passed to `proposal.sample()`.
        alternative_method: An alternative method for sampling from the restricted
            proposal. E.g., for SNPE, we suggest to sample with MCMC if the rejection
            rate is too high. Used only for printing during a potential warning.
        kwargs: Absorb additional unused arguments that can be passed to
            `rejection_sample()`. Warn if not empty.

    Returns:
        Accepted samples of shape `(sample_dim, batch_dim, *event_shape)`, and
        acceptance rates for each observation.
    """

    if kwargs:
        logging.warn(
            f"You passed arguments to `rejection_sampling_parameters` that "
            f"are unused when you do not specify a `proposal` in the same "
            f"dictionary. The unused arguments are: {kwargs}"
        )

    # Progress bar can be skipped, e.g. when sampling after each round just for
    # logging.
    pbar = tqdm(
        disable=not show_progress_bars,
        total=num_samples,
        desc=f"Drawing {num_samples} posterior samples",
    )
    if proposal_sampling_kwargs is None:
        proposal_sampling_kwargs = {}

    num_remaining = num_samples

    # NOTE: We might want to change this to a more general approach in the future.
    # Currently limited to a single "batch_dim" for the condition.
    # But this would require giving the method the condition_shape explicitly...
    if "condition" in proposal_sampling_kwargs:
        num_xos = proposal_sampling_kwargs["condition"].shape[0]
    else:
        num_xos = 1

    accepted = [[] for _ in range(num_xos)]
    acceptance_rate = torch.full((num_xos,), float("Nan"))
    leakage_warning_raised = False
    # Ruff suggestion

    # To cover cases with few samples without leakage:
    sampling_batch_size = min(num_samples, max_sampling_batch_size)
    num_sampled_total = torch.zeros(num_xos)
    num_samples_possible = 0

    # fix a constant MC Dropout mask seed so samples drawn across iterations
    # come from the same ensemble
    if mcd_seed is None:
        mcd_seed = np.random.randint(2**30)

    while num_remaining > 0:
        # Sample and reject.
        candidates = proposal.sample(
            (sampling_batch_size,),  # type: ignore
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
            **proposal_sampling_kwargs,
        )

        # SNPE-style rejection-sampling when the proposal is the neural net.
        are_accepted = accept_reject_fn(candidates)
        # Reshape necessary in certain cases which do not follow the shape conventions
        # of the "DensityEstimator" class.
        are_accepted = are_accepted.reshape(sampling_batch_size, num_xos)
        candidates_to_reject = candidates.reshape(
            sampling_batch_size, num_xos, *candidates.shape[candidates.ndim - 1 :]
        )

        for i in range(num_xos):
            accepted[i].append(candidates_to_reject[are_accepted[:, i], i])

        # Update.
        # Note: For any condition of shape (*batch_shape, *condition_shape), the
        # samples will be of shape(sampling_batch_size,*batch_shape, *event_shape)
        # and hence work in dim = 0.
        num_accepted = are_accepted.sum(dim=0)
        # print(f'Num accepted: {num_accepted}')
        num_sampled_total += num_accepted.to(num_sampled_total.device)
        # print(f'Num sampled total: {num_sampled_total}')
        # print(f'Num below quota: {(num_sampled_total<num_samples).sum().item()}')
        num_samples_possible += sampling_batch_size

        min_num_accepted = num_sampled_total.min().item()
        num_remaining = num_samples - min_num_accepted
        # Below are 2 orig SBI lines, but the logic seems wrong
        # min_num_accepted = num_accepted.min().item()
        # num_remaining -= min_num_accepted
        # pbar.update(min_num_accepted)
        pbar.update(min_num_accepted - pbar.n)

        # To avoid endless sampling when leakage is high, we raise a warning if the
        # acceptance rate is too low after the first 1_000 samples.
        acceptance_rate = num_sampled_total / num_samples_possible
        min_acceptance_rate = acceptance_rate.min().item()

        # For remaining iterations (leakage or many samples) continue
        # sampling with fixed batch size, reduced in cased the number
        # of remaining samples is low. The `max(..., 1e-12)` is to avoid division
        # by zero if acceptance rate is zero.
        sampling_batch_size = min(
            max_sampling_batch_size,
            max(int(1.5 * num_remaining / max(min_acceptance_rate, 1e-12)), 100),
        )
        if (
            num_sampled_total.min().item() > 1000
            and min_acceptance_rate < warn_acceptance
            and not leakage_warning_raised
        ):
            if sample_for_correction_factor:
                idx_min = acceptance_rate.argmin().item()
                logging.warning(
                    f"""Drawing samples from posterior to estimate the normalizing
                        constant for `log_prob()`. However, only
                        {min_acceptance_rate:.3%} posterior samples are within the
                        prior support (for condition {idx_min}). It may take a long time
                        to collect the remaining {num_remaining} samples.
                        Consider interrupting (Ctrl-C) and either basing the
                        estimate of the normalizing constant on fewer samples (by
                        calling `posterior.leakage_correction(x_o,
                        num_rejection_samples=N)`, where `N` is the number of
                        samples you want to base the
                        estimate on (default N=10000), or not estimating the
                        normalizing constant at all
                        (`log_prob(..., norm_posterior=False)`. The latter will
                        result in an unnormalized `log_prob()`."""
                )
            else:
                warn_msg = f"""Only {min_acceptance_rate:.3%} proposal samples are
                    accepted. It may take a long time to collect the remaining
                    {num_remaining} samples. """
                if alternative_method is not None:
                    warn_msg += f"""Consider interrupting (Ctrl-C) and switching to
                    `{alternative_method}`."""
                logging.warning(warn_msg)

            leakage_warning_raised = True  # Ensure warning is raised just once.

    pbar.close()

    # When in case of leakage a batch size was used there could be too many samples.
    samples = [torch.cat(accepted[i], dim=0)[:num_samples] for i in range(num_xos)]
    samples = torch.stack(samples, dim=1)
    samples = samples.reshape(num_samples, *candidates.shape[1:])
    assert (
        samples.shape[0] == num_samples
    ), "Number of accepted samples must match required samples."

    # NOTE: Restriction prior does currently require a float as return for the
    # acceptance rate, which is why we for now also return the minimum acceptance rate.

    # NOTE2 (custom): we override this behavior as we actually care about
    # batch-by-batch acceptance rates, which can later be associated with
    # component models (even if there are multiple batches/component)
    return samples, acceptance_rate  # as_tensor(min_acceptance_rate)

@torch.no_grad()
def accept_reject_sample_and_log_prob(
    proposal: nn.Module | Distribution,
    condition: Tensor,
    accept_reject_fn: Callable,
    num_samples: int,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    sample_for_correction_factor: bool = False,
    max_sampling_batch_size: int = 10_000,
    proposal_sampling_kwargs: dict | None = None,
    alternative_method: str | None = None,
    mcd_num_batches: int | None = None,
    mcd_seed: int | None = None,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    r"""Returns samples from a proposal according to a acception criterion.

    This is relevant for snpe methods and flows for which the posterior tends to have
    mass outside the prior support.

    This function could in principle be integrated into `rejection_sample()`. However,
    to keep the warnings clean, to avoid additional code for integration, and confusing
    if-cases, we decided to keep two separate functions.

    This function uses rejection sampling with samples from posterior in order to
        1) obtain posterior samples within the prior support, and
        2) calculate the fraction of accepted samples as a proxy for correcting the
           density during evaluation of the posterior.

    Args:
        posterior_nn: Neural net representing the posterior.
        accept_reject_fn: Function that evaluatuates which samples are accepted or
            rejected. Must take a batch of parameters and return a boolean tensor which
            indicates which parameters get accepted.
        num_samples: Desired number of samples.
        show_progress_bars: Whether to show a progressbar during sampling.
        warn_acceptance: A minimum acceptance rate under which to warn about slowness.
        sample_for_correction_factor: True if this function was called by
            `leakage_correction()`. False otherwise. Will be used to adapt the leakage
             warning and to decide whether we have to search for the maximum.
        max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.
        proposal_sampling_kwargs: Arguments that are passed to `proposal.sample()`.
        alternative_method: An alternative method for sampling from the restricted
            proposal. E.g., for SNPE, we suggest to sample with MCMC if the rejection
            rate is too high. Used only for printing during a potential warning.
        kwargs: Absorb additional unused arguments that can be passed to
            `rejection_sample()`. Warn if not empty.

    Returns:
        Accepted samples of shape `(sample_dim, batch_dim, *event_shape)`, and
        acceptance rates for each observation.
    """

    if kwargs:
        logging.warn(
            f"You passed arguments to `rejection_sampling_parameters` that "
            f"are unused when you do not specify a `proposal` in the same "
            f"dictionary. The unused arguments are: {kwargs}"
        )

    # Progress bar can be skipped, e.g. when sampling after each round just for
    # logging.
    pbar = tqdm(
        disable=not show_progress_bars,
        total=num_samples,
        desc=f"Drawing {num_samples} posterior samples",
    )

    num_remaining = num_samples

    # NOTE: We might want to change this to a more general approach in the future.
    # Currently limited to a single "batch_dim" for the condition.
    # But this would require giving the method the condition_shape explicitly...
    num_xos = condition.shape[0]

    accepted = [[] for _ in range(num_xos)]
    accepted_log_probs = [[] for _ in range(num_xos)]
    acceptance_rate = torch.full((num_xos,), float("Nan"))
    leakage_warning_raised = False
    # Ruff suggestion

    # To cover cases with few samples without leakage:
    sampling_batch_size = min(num_samples, max_sampling_batch_size)
    num_sampled_total = torch.zeros(num_xos)
    num_samples_possible = 0

    if not mcd_seed:
        mcd_seed = np.random.randint(2**30)

    while num_remaining > 0:
        # Sample and reject.
        candidates, log_probs = proposal.sample_and_log_prob(
            (sampling_batch_size,),  # type: ignore
            condition=condition,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )
        # SNPE-style rejection-sampling when the proposal is the neural net.
        are_accepted = accept_reject_fn(candidates)
        # Reshape necessary in certain cases which do not follow the shape conventions
        # of the "DensityEstimator" class.
        are_accepted = are_accepted.reshape(sampling_batch_size, num_xos)
        candidates_to_reject = candidates.reshape(
            sampling_batch_size, num_xos, *candidates.shape[candidates.ndim - 1 :]
        )
        log_probs_to_reject = log_probs.reshape(
            sampling_batch_size, num_xos, 1
        )

        for i in range(num_xos):
            accepted[i].append(candidates_to_reject[are_accepted[:, i], i])
            accepted_log_probs[i].append(log_probs_to_reject[are_accepted[:, i], i])

        # Update.
        # Note: For any condition of shape (*batch_shape, *condition_shape), the
        # samples will be of shape(sampling_batch_size,*batch_shape, *event_shape)
        # and hence work in dim = 0.
        num_accepted = are_accepted.sum(dim=0)
        num_sampled_total += num_accepted.to(num_sampled_total.device)
        num_samples_possible += sampling_batch_size

        min_num_accepted = num_sampled_total.min().item()
        num_remaining = num_samples - min_num_accepted
        pbar.update(min_num_accepted - pbar.n)

        # To avoid endless sampling when leakage is high, we raise a warning if the
        # acceptance rate is too low after the first 1_000 samples.
        acceptance_rate = num_sampled_total / num_samples_possible
        min_acceptance_rate = acceptance_rate.min().item()

        # For remaining iterations (leakage or many samples) continue
        # sampling with fixed batch size, reduced in cased the number
        # of remaining samples is low. The `max(..., 1e-12)` is to avoid division
        # by zero if acceptance rate is zero.
        sampling_batch_size = min(
            max_sampling_batch_size,
            max(int(1.5 * num_remaining / max(min_acceptance_rate, 1e-12)), 100),
        )
        if (
            num_sampled_total.min().item() > 1000
            and min_acceptance_rate < warn_acceptance
            and not leakage_warning_raised
        ):
            if sample_for_correction_factor:
                idx_min = acceptance_rate.argmin().item()
                logging.warning(
                    f"""Drawing samples from posterior to estimate the normalizing
                        constant for `log_prob()`. However, only
                        {min_acceptance_rate:.3%} posterior samples are within the
                        prior support (for condition {idx_min}). It may take a long time
                        to collect the remaining {num_remaining} samples.
                        Consider interrupting (Ctrl-C) and either basing the
                        estimate of the normalizing constant on fewer samples (by
                        calling `posterior.leakage_correction(x_o,
                        num_rejection_samples=N)`, where `N` is the number of
                        samples you want to base the
                        estimate on (default N=10000), or not estimating the
                        normalizing constant at all
                        (`log_prob(..., norm_posterior=False)`. The latter will
                        result in an unnormalized `log_prob()`."""
                )
            else:
                warn_msg = f"""Only {min_acceptance_rate:.3%} proposal samples are
                    accepted. It may take a long time to collect the remaining
                    {num_remaining} samples. """
                if alternative_method is not None:
                    warn_msg += f"""Consider interrupting (Ctrl-C) and switching to
                    `{alternative_method}`."""
                logging.warning(warn_msg)

            leakage_warning_raised = True  # Ensure warning is raised just once.

    pbar.close()

    # When in case of leakage a batch size was used there could be too many samples.
    samples = [torch.cat(accepted[i], dim=0)[:num_samples] for i in range(num_xos)]
    samples = torch.stack(samples, dim=1)
    samples = samples.reshape(num_samples, *candidates.shape[1:])
    assert (
        samples.shape[0] == num_samples
    ), "Number of accepted samples must match required samples."

    log_probs = [torch.cat(accepted_log_probs[i], dim=0)[:num_samples] for i in range(num_xos)]
    log_probs = torch.stack(log_probs, dim=1)
    log_probs = log_probs.reshape(num_samples, log_probs.shape[1]) # strip 3D
    assert (
        log_probs.shape[0] == num_samples
    ), "Number of accepted sample log-probs must match required samples."

    # NOTE: Restriction prior does currently require a float as return for the
    # acceptance rate, which is why we for now also return the minimum acceptance rate.
    return samples, log_probs, acceptance_rate  # as_tensor(min_acceptance_rate)
