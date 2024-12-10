from typing import Optional
from inspect import signature
import types
import logging

import numpy as np

from nflows import flows, transforms
from nflows.transforms import made as made_module
from nflows.utils import torchutils
import nflows.utils.typechecks as check

import torch
from torch import Tensor, nn, relu, tanh
from torch.nn import functional as F

from sbi.neural_nets.estimators import NFlowsFlow
from sbi.neural_nets.net_builders.flow import get_base_dist
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import (
    standardizing_net,
    standardizing_transform,
    z_score_parser,
)
from sbi.utils.user_input_checks import check_data_device
from sbi.sbi_types import Shape

from seqinf.dropout import ConsistentMCDropout
from seqinf.util import printd


logger = logging.getLogger(__name__)

nflow_specific_kwargs = ["num_bins", "num_components", "tail_bound"]


class BayesMaskedFeedforwardBlock(made_module.MaskedFeedforwardBlock):
    """A feedforward block based on a masked linear module.

    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """
    def __init__(
        self,
        *args,
        dropout_probability=0.0,
        **kwargs
    ):
        super().__init__(*args, dropout_probability=dropout_probability, **kwargs)

        self.mc_dropout = ConsistentMCDropout(p=dropout_probability)

    def forward(
        self,
        inputs,
        context=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        if self.batch_norm:
            temps = self.batch_norm(inputs)
        else:
            temps = inputs
        temps = self.linear(temps)
        temps = self.activation(temps)

        #outputs = self.dropout(temps)
        outputs = self.mc_dropout(
            temps,
            num_batches=mcd_num_batches,
            seed=mcd_seed
        )

        return outputs


class BayesMaskedResidualBlock(made_module.MaskedResidualBlock):
    """
    A residual block containing masked linear modules.
    """

    def __init__(
        self,
        *args,
        dropout_probability=0.0,
        **kwargs
    ):
        super().__init__(*args, dropout_probability=dropout_probability, **kwargs)

        self.mc_dropout = ConsistentMCDropout(p=dropout_probability)

    def forward(
        self,
        inputs,
        context=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if context is not None:
            temps += self.context_layer(context)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)

        #temps = self.dropout(temps)
        temps = self.mc_dropout(
            temps,
            num_batches=mcd_num_batches,
            seed=mcd_seed
        )

        temps = self.linear_layers[1](temps)
        return inputs + temps
        

class BayesMADE(made_module.MADE):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        output_multiplier=1,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super(made_module.MADE, self).__init__()

        # Initial layer.
        self.initial_layer = made_module.MaskedLinear(
            in_degrees=made_module._get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False,
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)

        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = BayesMaskedResidualBlock
        else:
            block_constructor = BayesMaskedFeedforwardBlock

        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = made_module.MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
        )

    def forward(
        self,
        inputs,
        context=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        # print(f'MCD mask seed (in MADE) :: {mcd_seed}')

        temps = self.initial_layer(inputs)
        if context is not None:
            temps += self.context_layer(context)

        for i, block in enumerate(self.blocks):
            temps = block(
                temps,
                context,
                mcd_num_batches=mcd_num_batches,
                mcd_seed=(
                    mcd_seed+i
                    if mcd_seed
                    else None
                )
            )

        outputs = self.final_layer(temps)

        return outputs


class BayesMaskedAffineAutoregressiveTransform(
    transforms.MaskedAffineAutoregressiveTransform
):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        self.features = features
        made = BayesMADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self._epsilon = 1e-3
        super(transforms.MaskedAffineAutoregressiveTransform, self).__init__(made)


    def forward(
        self,
        inputs,
        context=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        autoregressive_params = self.autoregressive_net(
            inputs,
            context=context,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed
        )
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(
        self,
        inputs,
        context=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        num_inputs = int(np.prod(inputs.shape[1:]))
        outputs = torch.zeros_like(inputs)
        logabsdet = None

        for i in range(num_inputs):
            autoregressive_params = self.autoregressive_net(
                outputs,
                context=context,
                mcd_num_batches=mcd_num_batches,
                mcd_seed=mcd_seed
            )
            outputs, logabsdet = self._elementwise_inverse(
                inputs, autoregressive_params
            )

        return outputs, logabsdet


class BayesCompositeTransform(transforms.Transform):
    '''
    Like typical CompositeTransform, but accept ``mcd_num_batches`` and
    optionally propagate them to the wrapped transforms, if they accept such a
    parameter.
    '''

    def __init__(self, transforms):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(
        inputs,
        funcs,
        context,
        forward: bool = True,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        funcs = list(funcs)

        num_transforms = len(list(funcs))
        seed_const = 1_000
        seed_list = [None]*num_transforms

        if mcd_seed:
            seed_list = [
                mcd_seed + (i * seed_const)
                for i in range(num_transforms)
            ]

        # reverse the seed list if doing an inverse pass
        if not forward:
            seed_list = seed_list[::-1]

        for i, func in enumerate(funcs):
            logger.info(
                f'Transform {i+1}/{num_transforms} [seed {seed_list[i]}] ' +
                f'in {"forward" if forward else "inverse"} cascade'
            )

            if forward:
                func_sig = signature(func.forward).parameters
            else:
                func_sig = signature(func).parameters

            f_kwargs = {}
            if 'mcd_num_batches' in func_sig:
                f_kwargs['mcd_num_batches'] = mcd_num_batches

            if 'mcd_seed' in func_sig:
                f_kwargs['mcd_seed'] = seed_list[i]

            outputs, logabsdet = func(outputs, context, **f_kwargs)
            total_logabsdet += logabsdet

        return outputs, total_logabsdet

    def forward(
        self,
        inputs,
        context=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        funcs = self._transforms

        return self._cascade(
            inputs, funcs, context,
            forward=True,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed
        )

    def inverse(
        self,
        inputs,
        context=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        funcs = (transform.inverse for transform in self._transforms[::-1])

        return self._cascade(
            inputs, funcs, context,
            forward=False,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed
        )


class BayesFlow(flows.Flow):
    '''
    Redefine most methods, keeping same overall structure but changing the
    ``self._transform`` calls.

    At the moment, this only appears to affect ``self._transform.inverse``
    calls, not forward passes.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sample(
        self,
        num_samples,
        context,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        '''
        NOTE: this method is only ever called by the parent ``sample()``, but
        ``context`` is allowed to be ``None``. It's odd that there's no
        handling of this, unless somehow ``_embedding_net`` will always turn a
        ``None`` into a tensor.

        Returns:
            (B, N, D)-shaped tensor containing batched samples
        '''
        embedded_context = self._embedding_net(context)

        if self._context_used_in_base:
            noise = self._distribution.sample(num_samples, context=embedded_context)
        else:
            # flatten the context and sample without the kwarg, then split back
            repeat_noise = self._distribution.sample(num_samples*embedded_context.shape[0])
            noise = torch.reshape(
                repeat_noise,
                (embedded_context.shape[0], -1, repeat_noise.shape[1])
            )  # should have shape (B, N, D)

        if embedded_context is not None:
            # should have 3+ dims
            assert len(noise.shape) >= 3

            if mcd_num_batches is None:
                # draw samples downstream consistently per batch
                mcd_num_batches = noise.shape[0]

            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform.inverse(
            noise,
            context=embedded_context,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def _log_prob(
        self,
        inputs,
        context,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        # print(f'MCD mask seed (in BayesFlow._log_prob) :: {mcd_seed}')

        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(
            inputs,
            context=embedded_context,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )

        if self._context_used_in_base:
            log_prob = self._distribution.log_prob(noise, context=embedded_context)
        else:
            log_prob = self._distribution.log_prob(noise)

        return log_prob + logabsdet

    def sample(
        self,
        num_samples,
        context=None,
        batch_size=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        """Generates samples from the distribution. Samples can be generated in batches.

        NOTE: this originates from the ``nflows.Distribution`` base; it's not
        defined at the ``flows.Flow`` level.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored. 
                     Should have shape [context_size, ...], where ... represents a (context) feature 
                     vector of arbitrary shape. This will generate num_samples for each context item 
                     provided. The overall shape of the samples will then be 
                     [context_size, num_samples, ...].
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given, where ... represents a feature
            vector of arbitrary shape.
        """
        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        mcd_kwargs = dict(
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context, **mcd_kwargs)

        else:
            if not check.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [
                self._sample(batch_size, context, **mcd_kwargs)
                for _ in range(num_batches)
            ]

            if num_leftover > 0:
                samples.append(
                    self._sample(num_leftover, context, **mcd_kwargs)
                )

            # samples come in shape (B, M, D), where M < N are sample batches
            # so transpose (aligning MC batches B), cat, untranspose
            samples = [s.transpose(0, 1) for s in samples]
            samples = torch.cat(samples, dim=0)
            samples = samples.transpose(0, 1)

            # output is full shape (B, N, D)
            return samples

    def log_prob(
        self,
        inputs,
        context=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        """Calculate log probability under the distribution.

        NOTE: this originates from the ``nflows.Distribution`` base; it's not
        defined at the ``flows.Flow`` level.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)

        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of context items."
                )

        return self._log_prob(
            inputs,
            context,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )

    def sample_and_log_prob(
        self,
        num_samples,
        context=None,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ):
        '''
        Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.

        NOTE: when context is specified, we typically expect a batched
        dimension, e.g., having shape (B, C). Call E the embedding dimension,
        D the underlying distribution dimension, and N the number of samples.
        Then:

        - Noise will have shape (B, N, D)
        - Noise flattens the first two dims into (B*N, D) before passing to
          ``self._transform.inverse``. This method *does not* support a batch
          dimension; failing to flatten out will cause the (typically)
          autoregressive transform steps to misinterpret the data. That is, it
          models dimensions [1:] of the passed tensor as the feature
          dimensions of the elements of dim [0], and the presence of a batch
          dimension will thus cause chaos.

          Note that in some cases we may actually have more than just a vector
          for each point (e.g., a 2D image), but the point about batches still
          applies, and the autoregressive component will flatten out those [1:]
          dimensions to treat them all as features.
        - Flattening the batch dimension, however, prevents us from being able
          to draw samples under the same underlying states of model components
          (e.g., for dropout). That is, I have no way of knowing which of
          samples that are being handled in parallel are part of which dropout
          batch, and cannot freeze the mask for some portion of those inputs.
          So we pass batch indices everywhere down the chain, only to ever have
          an effect when an implementing block does something with (e.g., in a
          ConsistentMCDropout layer). Otherwise, we continue to process
          points/features the same as before, keeping the same shape.
        '''
        embedded_context = self._embedding_net(context)
        if self._context_used_in_base:
            noise, log_prob = self._distribution.sample_and_log_prob(
                num_samples, context=embedded_context
            )  # will have shape (B, N, D) if context not None
        else:
            noise, log_prob = self._distribution.sample_and_log_prob(
                num_samples
            )  # will have shape (N, D)

        if embedded_context is not None:
            # should have 3+ dims
            assert len(noise.shape) >= 3

            if mcd_num_batches is None:
                # draw samples downstream consistently per batch
                mcd_num_batches = noise.shape[0]

            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(
            noise,
            context=embedded_context,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise


class BayesNFlowsFlow(NFlowsFlow):
    '''
    Exactly the same as ``sbi's`` ``NFlowsFlow``, but fixes the
    ``sample_and_log_prob`` method to match the output structure seen in
    ``sample``. My best guess is that the latter was updated recently but the
    former was left behind (although the issue does appear to date back to the
    last actual release). Nevertheless, ``sample_and_log_prob`` does *not*
    transpose the first two dimensions, swapping the "sample shape" and "batch
    dimension," as we do in ``sample()``. So we do that here.

    We also redefine ``log_prob`` to accept ``mcd_num_batches`` from the
    outermost level.

    NOTE: there's a chance this may not matter outside of the consistent MC
    dropout use case, but I can't imagine it's all too great that the outputs
    don't align.
    '''

    def sample(
        self,
        sample_shape: Shape,
        condition: Tensor,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(sample_dim, batch_dim, *event_shape)`.

        Returns:
            Samples of shape `(*sample_shape, condition_batch_dim)`.
        """
        condition_batch_dim = condition.shape[0]
        num_samples = torch.Size(sample_shape).numel()

        samples = self.net.sample(
            num_samples,
            context=condition,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed
        )
        # Change from Nflows' convention of (batch_dim, sample_dim, *event_shape) to
        # (sample_dim, batch_dim, *event_shape) (PyTorch + SBI).
        samples = samples.transpose(0, 1)

        return samples.reshape((*sample_shape, condition_batch_dim, *self.input_shape))

    def log_prob(
        self,
        input: Tensor,
        condition: Tensor,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
    ) -> Tensor:
        r"""Return the log probabilities of the inputs given a condition or multiple
        i.e. batched conditions.

        Args:
            input: Inputs to evaluate the log probability on. Of shape
                `(sample_dim, batch_dim, *event_shape)`.
            condition: Conditions of shape `(sample_dim, batch_dim, *event_shape)`.

        Raises:
            AssertionError: If `input_batch_dim != condition_batch_dim`.

        Returns:
            Sample-wise log probabilities, shape `(input_sample_dim, input_batch_dim)`.
        """
        input_sample_dim = input.shape[0]
        input_batch_dim = input.shape[1]
        condition_batch_dim = condition.shape[0]
        condition_event_dims = len(condition.shape[1:])

        assert condition_batch_dim == input_batch_dim, (
            f"Batch shape of condition {condition_batch_dim} and input "
            f"{input_batch_dim} do not match."
        )

        # BELOW IS THE ORIGINAL SBI RESHAPING
        # WE ARE CHANGING THIS TO USE A LEADING BATCH DIM FOR INTERNAL OPS

        # # Nflows needs to have a single batch dimension for condition and input.
        # input = input.reshape((input_batch_dim * input_sample_dim, -1))

        # # Repeat the condition to match `input_batch_dim * input_sample_dim`.
        # ones_for_event_dims = (1,) * condition_event_dims  # Tuple of 1s, e.g. (1, 1, 1)
        # condition = condition.repeat(input_sample_dim, *ones_for_event_dims)

        # input is shape (N, B, D); transpose to (B, N, D)
        input = input.transpose(0, 1)

        # then flatten such that batches are "stacked"
        #input = input.reshape((input_batch_dim * input_sample_dim, -1))
        input = torchutils.merge_leading_dims(input, num_dims=2)

        # stretch the condition rows rather than stacking to match the leading
        # dim of input and keep row-by-row batch affiliation
        condition = torchutils.repeat_rows(
            condition, num_reps=input_sample_dim
        )

        log_probs = self.net.log_prob(
            input,
            context=condition,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )  # returns size (N*B, 1)

        # reshape to (B, N, D)
        log_probs = log_probs.reshape((input_batch_dim, input_sample_dim))

        # then transpose to (N, B, D) for SBI standards
        log_probs = log_probs.transpose(0, 1)

        return log_probs

    def sample_and_log_prob(
        self,
        sample_shape: torch.Size,
        condition: Tensor,
        mcd_num_batches: int | None = None,
        mcd_seed: int | None = None,
        **kwargs
    ) -> tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape (sample_dim, batch_dim, *event_shape).

        Returns:
            Samples of shape `(*sample_shape, condition_batch_dim, *input_event_shape)`
            and associated log probs of shape `(*sample_shape, condition_batch_dim)`.
        """
        condition_batch_dim = condition.shape[0]
        num_samples = torch.Size(sample_shape).numel()

        samples, log_probs = self.net.sample_and_log_prob(
            num_samples,
            context=condition,
            mcd_num_batches=mcd_num_batches,
            mcd_seed=mcd_seed,
        )

        #samples = samples.reshape((*sample_shape, condition_batch_dim, -1))
        #log_probs = log_probs.reshape((*sample_shape, -1))

        samples = samples.transpose(0, 1)
        log_probs = log_probs.transpose(0, 1)

        samples = samples.reshape((*sample_shape, condition_batch_dim, *self.input_shape))
        log_probs = log_probs.reshape((*sample_shape, condition_batch_dim))

        return samples, log_probs


def build_bmaf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
) -> NFlowsFlow:
    """Builds MAF p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        num_blocks: number of blocks used for residual net for context embedding.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(
        batch_x,
        embedding_net=None,
        warn_on_1d=True,  # warn if output space is 1D.
    )
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    transform_list = []
    for _ in range(num_transforms):
        block = [
            BayesMaskedAffineAutoregressiveTransform(
                features=x_numel,
                hidden_features=hidden_features,
                context_features=y_numel,
                num_blocks=num_blocks,
                use_residual_blocks=False,
                random_mask=False,
                activation=tanh,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ),
            transforms.RandomPermutation(features=x_numel),
        ]
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # Combine transforms
    transform = BayesCompositeTransform(transform_list)

    distribution = get_base_dist(x_numel, **kwargs)
    neural_net = BayesFlow(transform, distribution, embedding_net)
    #neural_net = flows.Flow(transform, distribution, embedding_net)
    #flow = NFlowsFlow(
    flow = BayesNFlowsFlow(
        neural_net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape
    )

    return flow
