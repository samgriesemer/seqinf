from inspect import signature
import logging

import numpy as np

from nflows import flows, transforms

import torch
from torch import nn
from torch.nn import functional as F

from seqinf.transforms.made import BMADE

logger = logging.getLogger(__name__)


class BMaskedAffineAutoregressiveTransform(
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
        made = BMADE(
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


class BCompositeTransform(transforms.Transform):
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
