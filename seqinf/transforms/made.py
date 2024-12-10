import logging

from nflows.transforms import made as made_module

from torch import nn
from torch.nn import functional as F

from seqinf.dropout import ConsistentMCDropout


logger = logging.getLogger(__name__)

class BMaskedFeedforwardBlock(made_module.MaskedFeedforwardBlock):
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


class BMaskedResidualBlock(made_module.MaskedResidualBlock):
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
        

class BMADE(made_module.MADE):
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
            block_constructor = BMaskedResidualBlock
        else:
            block_constructor = BMaskedFeedforwardBlock

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
