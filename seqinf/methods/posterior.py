from typing import Callable
from copy import deepcopy

from torch import Tensor, nn
from torch.distributions import Distribution

from sbi.neural_nets import posterior_nn

import sbibm
from sbibm.tasks import Task

from seqinf.flow import build_bmaf
from seqinf.util import prepare_sbibm_prior
from seqinf.collector import (
    BayesianCollector,
    ASNPECollector,
    ActiveCollector,
    BatchActiveCollector
)
from seqinf.sequential import SequentialInference
from seqinf.posterior import BayesDirectPosterior


class SNPE(SequentialInference):
    '''
    Sequential NPE base

    A bare bones invocation may look like

    .. code-block:: python

        SNPE(simulator, prior, num_workers=-1)


    where ``num_workers=-1`` will attempt to batch simulator runs over all
    available cores.
    '''

    def __init__(
        self,
        simulator: Callable,
        prior: Distribution | None = None,
        density_estimator: str | Callable | None = 'maf',
        de_kwargs: dict | None = None,
        embedding_net: nn.Module = nn.Identity(),

        sample_with: str | None = 'direct',
        sample_method: str | None = None,
        sample_parameters: dict | None = None,
        proposal_method: str | None = 'posterior',

        **kwargs
    ):
        super().__init__(
            method='NPE_C',
            simulator=simulator,
            prior=prior,
            density_estimator=density_estimator,
            de_kwargs=de_kwargs,
            embedding_net=embedding_net,

            sample_with=sample_with,
            sample_method=sample_method,
            sample_parameters=sample_parameters,
            proposal_method=proposal_method,

            **kwargs
        )

    @classmethod
    def from_task(
        cls,
        task_name: str,
        **kwargs,
    ):
        '''
        (have to redefine alternate constructors as well)
        '''
        task = sbibm.get_task(task_name)
        assert task is not None

        simulator = task.get_simulator()
        prior = task.prior_dist

        return cls(
            simulator=simulator,
            prior=prior,
            task=task,
            **kwargs
        )

    def _density_builder_wrapper(self) -> Callable:
        return posterior_nn(
            model=self.density_estimator,
            embedding_net=self.embedding_net,
            **self.de_kwargs,
        )


class BayesianSNPE(SNPE):
    collector_cls = BayesianCollector

    def __init__(
        self,
        *args,
        de_kwargs: dict | None = dict(dropout_probability=0.5),
        **kwargs,
    ):
        super().__init__(
            *args,
            de_kwargs=de_kwargs,
            **kwargs,
        )

    def _density_builder_wrapper(self) -> Callable:
        def build_fn(batch_theta, batch_x):
            return build_bmaf(
                batch_x=batch_theta,
                batch_y=batch_x,
                embedding_net=self.embedding_net,
                **self.de_kwargs
            )

        return build_fn

    def _build_proposal(self, x_o: Tensor = None):
        '''
        Mimics ``sbi's`` ``PosteriorEstimator.build_posterior()`` method to
        allow a custom ``NeuralPosterior`` wrapper. The method in the ``sbi``
        package only accepts a string argument for ``sample_with``, after which
        it constructs an appropriate wrapper (e.g., ``DirectPosterior``) on the
        density estimator. In this case, none of these built-in options are
        suitable, as I need ``.sample_and_log_prob()`` implemented at the
        level of this wrapper (and potentially some specifics for correcting
        leakage in BNFs).

        NOTE: for ``device``, see how it's typically handled in a place like
        ``sbi.npe_base``. The device is specified at the ``NeuralInference``
        level (e.g., NPE-C, which corresponds to the ``self.method_func`` call
        in ``SequentialInference.reset()``), and when the DE is created (usually
        in ``train()``), it is first done so on the CPU and *then* moved to the
        target device. Then, in ``build_posterior()``, the device is inferred
        based on where the DE is, and passed on to a relevant wrapped like
        ``DirectPosterior``. This method replaces the logic found in ``sbi's``
        ``build_posterior()``, so we need to handle devices properly. We expect
        the DE, which is still created and managed by the same ``sbi`` logic
        when we call ``train()``, *should* have the correct device
        automatically. Here we simply peak at the DE model and check anyway,
        verifying that constraint holds with an assertion and passing it on to
        our own ``DirectPosterior`` child.
        '''
        # prior is OK
        # nde is OK
        device = str(next(self.trained_de.parameters()).device)

        # device checks are typically handled by `sbi` internally as it creates
        # posterior wrappers. In this case the above device should *still* be
        # handled correctly by `sbi`, but we double check this matches.
        assert device == self.device

        _posterior = BayesDirectPosterior(
            posterior_estimator=self.trained_de,
            prior=self.prior,
            device=device,
            **self.sample_parameters or {},
        )

        # manually populate inference vars with deep copied posterior
        self.inference._posterior = _posterior
        self.inference._model_bank.append(deepcopy(_posterior))
        posterior = deepcopy(_posterior)

        if x_o is not None:
            posterior = posterior.set_default_x(x_o)

        return self._apply_proposal_method(posterior)


class ASNPE(BayesianSNPE):
    '''
    ASNPE wrapper (NeurIPS 2412.05590)
    '''
    collector_cls = ASNPECollector
