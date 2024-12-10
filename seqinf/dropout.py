import torch
from torch import nn, Tensor


class ConsistentMCDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError(
                f'dropout prob. has to be between 0 and 1 but got {p}'
            )

        self.p = p
        self.mask = None

        # should be used after explicitly setting seed
        self.mask_gen = torch.Generator()

    def extra_repr(self):
        return f'p={self.p}'

    def train(self, mode=True):
        # make sure self.training is set
        super().train(mode)

        if not mode:
            self.reset_mask()

    def reset_mask(self):
        self.mask = None

    def _generate_dropout_mask(
        self,
        shape,
        device = None,
        seed: int | None = None
    ):
        '''
        Generates a dropout mask for a given shape.

        Parameters:
            shape: shape of mask tensor (typically (B, 1, D))
            device: device to use for ``torch.rand``
            seed: random seed for the ``torch.rand`` generator
        '''
        if seed is not None:
            self.mask_gen.manual_seed(seed)

            return (
                torch.rand(
                    shape,
                    device=device,
                    generator=self.mask_gen
                ) > self.p
            ).float()

        return (torch.rand(shape, device=device) > self.p).float()

    def _apply_dropout(self, input, dropout_mask):
        '''
        Applies a dropout mask and scales the remaining elements.

        Parameters:
            input: (B, N, D)-shaped input tensor
            dropout_mask: (B, 1, D)-shaped dropout mask
        '''
        output = input * dropout_mask

        return output / (1 - self.p)

    def forward(
        self,
        input: Tensor,
        num_batches: int | None = None,
        seed: int | None = None,
    ):
        '''
        Expects ``input`` to be shaped (N, D*), where D* may be several
        dimensions (D1, D2, ..., Dm) for features at each of the N points
        (e.g., 2-dimensional if we had N images). Note how there's no batch
        dimension out front, e.g., (B, N, D*). This is b/c along the flow
        chain, the batch dimension gets flattened to play nice with Transform
        methods (forward and inverse). It stays that way until it gets here, at
        which point we can unflatten it using the ``num_batches`` and apply a
        fixed dropout mask to each batch.

        Note that, if providing ``num_batches``, the input should be stacked
        accordingly. That is, if the number of intended batches is B, the first
        N/B input rows (``input[:(N/B)]``) will be used for the first batch,
        the second N/B rows (``input[(N/B):(2*N/B)]``) for the second batch,
        and so on.

        Parameters:
            input: input tensor to be processed, shaped (N, D*)
            num_batches: the number of batches (B) to use when reshaping the
                input. Must be an integer divisor of N. ``input`` will be
                reshaped to (N/B, B, D), where batches are simple splits from
                the N input rows. If unspecified, the number of batches will be
                set to N, and each input row will be seen as its own "batch"
                that receives its own mask.
            seed: seed to use when generating the dropout mask. This
                facilitates reproducible draws/evaluations from MC ensemble
                components.
        '''
        # print(f'MCD mask seed (in dropout) :: {seed}')

        if self.p == 0.0 or not self.training:
            return input

        if num_batches is None:
            num_batches = input.shape[0]

        N = input.shape[0] 
        B = num_batches
        M = N // B
        D = input.shape[1:]

        # print(f'Dropout shapes: {N}, {B}, {M}, {D}')

        # resulting batch size N/B must be an integer
        assert N % B == 0

        unflattened_input = input.view(B, M, *D)
        dropout_mask = self._generate_dropout_mask(
            (B, 1, *D),
            input.device,
            seed
        )
        output = self._apply_dropout(unflattened_input, dropout_mask)

        # print(f'B={B} N={N} D={D[0]} :: batch {batch_indices} :: mask {dropout_mask}')

        return output.view(-1, *D)
