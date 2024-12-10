'''
Sequential inference utility functions

Note on simulation helpers:

It appears ``sbi's`` ``simulate_in_batches`` and/or ``simulate_for_sbi`` are
currently in a state of flux (as of 0.23.1). In particular, the
``process_simulator`` method has traditionally relied on
``wrap_as_joblib_efficient_simulator`` (and still does, apparently). But there
is some inefficient Torch to Numpy casting before batching with joblib, and
while this appears to have been partially addressed with #1175, it looks like
it's mostly marginal and for compatibility with new joblib versions. As I
understand, there's still some significant inefficiencies to be fixed in later
PRs, but are not present in the current version.

In any case, the new compatibility changes are only for ``simulate_for_sbi``
and not the slimmer batch helper (which looks like will be deprecated in the
future). As such, I here strip out the underlying batch component of the "new"
``simulate_for_sbi`` method so I'm not forced to sample from the proposal
directly.
'''
import os
import re
import logging
import textwrap
from typing import Any, Callable, Tuple

import numpy as np
import torch
from joblib import Parallel, delayed
from numpy import ndarray
from torch import Tensor, float32
from tqdm.auto import tqdm

from sbi.utils.sbiutils import seed_all_backends


logger = logging.getLogger(__name__)


def simulate_in_batches(
    simulator: Callable,
    theta: Tensor,
    num_workers: int = 1,
    simulation_batch_size: int | None = 1,
    seed: int | None = None,
    show_progress_bar: bool = True,
) -> Tensor:
    '''
    Run the simulator in batches for provided parameters.

    This is exactly the source for ``simulate_for_sbi``, but with the
    ``proposal`` arg swapped out for explicit ``theta`` (and the
    ``num_simulations`` is thus inferred). That's it; very basic/minor changes
    to accommodate this.

    Parameters:
        simulator: A function that takes parameters $\theta$ and maps them to
            simulations, or observations, `x`, $\text{sim}(\theta)\to x$. Any
            regular Python callable (i.e. function or class with `__call__`
            method) can be used. Note that the simulator should be able to
            handle numpy arrays for efficient parallelization. You can use
            `process_simulator` to ensure this.
        theta: Parameters at which to run the simulator. Should be a tensor
            with a batch dimension (e.g., drawn like ``prior.sample((N,))``)
        num_workers: Number of parallel workers to use for simulations. -1 will
            attempt to use all available CPUs (expanded manually rather than
            passing to joblib)
        simulation_batch_size: Size of the batches to pass to the simulator per
            call. The provided ``theta`` will be split into ``N / batch_size``
            batches and distributed among the ``num_workers`` workers. If None,
            the batch size will be set to ``N`` (the full size of the passed
            theta), i.e., we simulate all parameter sets with one call,
            although by default we set the batch size to 1 and don't
            parallelize runs. If set to -1, we attempt to divvy up the number
            of samples across the ``num_workers`` (up to some maximum).
        seed: Seed for reproducibility.
        show_progress_bar: Whether to show a progress bar for simulating. This
            will not affect whether there will be a progressbar while drawing
            samples from the proposal.
    '''
    num_simulations, *_ = theta.shape
    device = theta.device

    if num_workers < 0:
        num_workers = os.cpu_count()

    if num_simulations == 0:
        x = torch.tensor([], dtype=float32)

    else:
        # Cast theta to numpy for better joblib performance (see #1175)
        seed_all_backends(seed)

        # Parse the simulation_batch_size logic
        if simulation_batch_size is None:
            # one single batch
            simulation_batch_size = num_simulations
        elif simulation_batch_size < 0:
            # implies num_batches = num_workers
            simulation_batch_size = max(num_simulations // num_workers, 1)
        else:
            simulation_batch_size = min(simulation_batch_size, num_simulations)

        if num_workers != 1:
            # For multiprocessing, we want to switch to numpy arrays.
            # The batch size will be an approximation, since np.array_split does
            # not take as argument the size of the batch but their total.
            num_batches = num_simulations // simulation_batch_size
            #batches = np.array_split(theta.numpy(), num_batches, axis=0)
            batches = np.array_split(theta.cpu().numpy(), num_batches, axis=0)
            batch_seeds = np.random.randint(low=0, high=1_000_000, size=(len(batches),))

            logger.info(
                f'Simulating {num_simulations} samples in {num_batches} batches '
                f'(size {simulation_batch_size}) with {num_workers} workers'
            )

            # define seeded simulator.
            def simulator_seeded(theta: ndarray, seed: int) -> Tensor:
                seed_all_backends(seed)
                return simulator(theta)

            try:  # catch TypeError to give more informative error message
                simulation_outputs: list[Tensor] = [  # pyright: ignore
                    xx
                    for xx in tqdm(
                        Parallel(return_as="generator", n_jobs=num_workers)(
                            delayed(simulator_seeded)(batch, seed)
                            for batch, seed in zip(batches, batch_seeds)
                        ),
                        total=num_simulations,
                        disable=not show_progress_bar,
                        leave=False
                    )
                ]
            except TypeError as err:
                raise TypeError(
                    "For multiprocessing, we switch to numpy arrays. Make sure to "
                    "preprocess your simulator with `process_simulator` to handle numpy"
                    " arrays."
                ) from err

        else:
            simulation_outputs: list[Tensor] = []
            batches = torch.split(theta, simulation_batch_size)
            for batch in tqdm(batches, disable=not show_progress_bar, leave=False):
                simulation_outputs.append(simulator(batch))

        # Correctly format the output
        x = torch.cat(simulation_outputs, dim=0)
        #theta = torch.as_tensor(theta, dtype=float32)

    x = x.to(device)

    return x


def printd(txt, indent=0, strip_nl=True):
    dedent_txt = textwrap.dedent(txt)

    if strip_nl:
        dedent_txt = re.sub(r'^\n?', '', dedent_txt, count=1)
        dedent_txt = re.sub(r'\n?$', '', dedent_txt, count=1)

    print(textwrap.indent(dedent_txt, ' '*indent))


def prepare_sbibm_prior(task):
    prior = task.get_prior()

    class WrappedPrior:
        @staticmethod
        def sample(shape):
            return prior(shape[0])

    return WrappedPrior()
