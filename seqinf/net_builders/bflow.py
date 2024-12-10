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

from seqinf.transforms.bflow import (
    BCompositeTransform,
    BMaskedAffineAutoregressiveTransform,
)
from seqinf.estimators.nflows_bflow import (
    BFlow,
    BNFlowsFlow,
)


logger = logging.getLogger(__name__)


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
            BMaskedAffineAutoregressiveTransform(
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
    transform = BCompositeTransform(transform_list)

    distribution = get_base_dist(x_numel, **kwargs)
    neural_net = BFlow(transform, distribution, embedding_net)
    #neural_net = flows.Flow(transform, distribution, embedding_net)
    #flow = NFlowsFlow(
    flow = BNFlowsFlow(
        neural_net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape
    )

    return flow
