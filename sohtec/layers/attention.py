from torch import nn


class Attention(nn.Module):
    """More efficient implementation of the attention layer using PyTorch's MultiheadAttention."""

    def __init__(
        self,
        dim,
        num_heads=16,
        bias=False,
        attn_drop=0.0,
    ):
        """Attention module.
        (B, N, D) -> (B, N, D)

        Args:
            dim (int): Input dimension.
            num_heads (int, optional): Defaults to 16.
            bias (bool, optional): Defaults to False.
            attn_drop (float, optional): Defaults to 0.0.
        """
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            bias=bias,
            add_bias_kv=False,
            dropout=attn_drop,
            batch_first=True,
        )

    def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (B, N, D), mask: (B, N). True for masked, False for unmasked.
        Return:
            x: (B, N, D)
            attn: (B, num_heads, N, N)
        """
        x, attn = self.self_attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        return x, (attn.detach() if need_weights else attn)
