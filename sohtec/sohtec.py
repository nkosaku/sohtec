from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .layers import MultiLayerPatchEmbed, TransformerBlock


@dataclass
class SOHTECBackboneConfig:
    seq_len: int = 1800
    patch_size: int = 4
    stride: int = 2
    in_channels: int = 4
    embed_dim: int = 256
    depth: int = 4
    n_heads: int = 16
    mlp_ratio: float = 4.0
    attn_bias: bool = False
    pos_drop_rate: float = 0.0
    ffn_drop_rate: float = 0.0
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.1
    norm_layer: nn.Module = nn.LayerNorm
    add_cls: bool = True
    post_norm: bool = False
    n_embed_layer: int = 2


class SOHTECBackbone(nn.Module):
    def __init__(
        self,
        config: Optional[SOHTECBackboneConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if config is None:
            config = SOHTECBackboneConfig(**kwargs)

        self.config = config
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.embed_dim = config.embed_dim
        self.add_cls = config.add_cls

        self.patch_embed = MultiLayerPatchEmbed(
            seq_len=config.seq_len,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            k1=self.patch_size,
            s1=self.stride,
            l=config.n_embed_layer,
        )
        npatch = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, npatch + 1 if config.add_cls else npatch, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.pos_drop_rate)

        dpr = [config.drop_path_rate * i / config.depth for i in range(config.depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.embed_dim,
                    num_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_bias=config.attn_bias,
                    drop=config.ffn_drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=config.norm_layer,
                    pre_norm=not config.post_norm,
                )
                for i in range(config.depth)
            ]
        )
        self.norm = config.norm_layer(config.embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.2)
        nn.init.trunc_normal_(self.cls_token, std=0.2)

    def prepare_tokens(
        self,
        x: torch.Tensor,
        mask_token_ratio: Optional[float] = None,
        mask_token: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare tokens and mask tokens before feeding them to the transformer.
        Patch embedding -> Add CLS token -> Add positional encoding

        Args:
            x: (B, C, L). Input tensor.
            mask_token_ratio: float. ratio of tokens to mask.
            mask_token: torch.Tensor. (D,). Token to replace the masked tokens.
            key_padding_mask: (B, L). key padding mask. True for mask, False for non-mask.

        Returns:
            z: (B, P, D). tokens after patch embed.
            target: (B, P, D). Logits after patch embed, used as target tokens for mask prediction.
                None if mask_token_ratio is None.
            mask: (B, P). True for mask, False for non-mask. None if mask_token_ratio is None.
            key_padding_mask_patched: (B, P). key padding mask. True for mask, False for non-mask.
        """
        B, C, L = x.shape

        assert C == self.config.in_channels
        assert L == self.config.seq_len

        z, key_padding_mask_patched = self.patch_embed(x, mask=key_padding_mask)

        # add the [CLS] token to the embed patch tokens
        if self.add_cls:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            z = torch.cat((cls_tokens, z), dim=1)

        # handle key padding mask
        if key_padding_mask_patched is not None:
            if self.add_cls:
                cls_valid = torch.zeros(
                    (B, 1), dtype=key_padding_mask_patched.dtype, device=key_padding_mask_patched.device
                )
                key_padding_mask_patched = torch.cat((cls_valid, key_padding_mask_patched), dim=1)

            key_padding_mask_patched = key_padding_mask_patched.to(dtype=torch.bool)

        B, P, D = z.shape

        # replace tokens with mask tokens
        if mask_token_ratio is not None:
            if mask_token is None:
                raise ValueError("mask_token must be provided if mask_token_ratio is not None")
            mask_token = mask_token.to(device=z.device, dtype=z.dtype)

            target = z.detach().clone()  # Tokens without masking.
            mask = torch.rand(B, P, device=z.device, dtype=torch.float32) < mask_token_ratio

            if key_padding_mask_patched is not None:
                mask[key_padding_mask_patched] = False

            if self.add_cls:
                mask[:, 0] = False  # do not mask the CLS token

            z[mask] = mask_token

        # add positional encoding to each token
        z = z + self.pos_embed  # (B, P, D)

        if mask_token_ratio is not None:
            return self.pos_drop(z), target, mask, key_padding_mask_patched
        return self.pos_drop(z), None, None, key_padding_mask_patched

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        mask_token_ratio: Optional[float] = None,
        mask_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return the output of the last block.
        args:
            x: (B, C, L)
            key_padding_mask: (B, L). key padding mask. True for mask, False for non-mask.
            mask_token_ratio: float. ratio of tokens to mask.
            mask_token: torch.Tensor. (D,). Token to replace the masked tokens.

        return:
            z: (B, P, D). output of the last block.
            target: (B, P, D). Logits after patch embed, used as target tokens for mask prediction.
                None if mask_token_ratio is None.
            mask: (B, P). True for mask, False for non-mask. None if mask_token_ratio is None.
        """
        z, target, mask, key_padding_mask_patched = self.prepare_tokens(
            x, mask_token_ratio, mask_token, key_padding_mask
        )

        for blk in self.blocks:
            z, _ = blk(z, key_padding_mask=key_padding_mask_patched)

        z = self.norm(z)

        return z, target, mask


class RegressionWrapper(nn.Module):
    def __init__(
        self,
        model: SOHTECBackbone,
        num_classes: int = 1,
        drop: float = 0.0,
        initial_bias: float = 75.0,
    ) -> None:
        super().__init__()
        self.model = model
        d = model.embed_dim

        self.initial_bias = initial_bias

        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Dropout(drop), nn.Linear(d // 2, num_classes))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.head[3].bias, self.initial_bias)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.model(x, key_padding_mask=key_padding_mask)[0]
        if self.model.add_cls:
            x = x[:, 0]  # Use the CLS token for regression
        else:
            raise NotImplementedError("RegressionWrapper does not support add_cls=False")
        return self.head(x)


class MaskPredWrapper(nn.Module):
    def __init__(
        self,
        model: SOHTECBackbone,
        mask_ratio: float = 0.5,
        drop: float = 0.0,
        loss_function: str = "MSE",
    ) -> None:
        super().__init__()
        self.model = model
        d = model.embed_dim

        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(d))

        self.regressor = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(d, d),
        )

        if loss_function == "MAE":
            self.criterion = nn.L1Loss()
        elif loss_function == "MSE":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss function: {loss_function}")

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z, target, mask = self.model(
            x,
            key_padding_mask=key_padding_mask,
            mask_token_ratio=self.mask_ratio,
            mask_token=self.mask_token,
        )
        z = self.regressor(z)
        z_masked = z[mask]
        target = target[mask]
        loss = self.criterion(z_masked, target)
        return loss


class SohCompareWrapper(nn.Module):
    def __init__(self, model: SOHTECBackbone, drop: float = 0.0) -> None:
        super().__init__()
        self.model = model
        d = model.embed_dim

        self.regressor = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(d, 1),
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        x: torch.Tensor,
        x2: torch.Tensor,
        target: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Output the SOH comparison result.
        The model outputs the probability of the first sample's SOH higher than the second sample's SOH.

        Args:
            x: (B, C, L). First sample.
            x2: (B, C, L). Second sample.
            target: (B,). Target label. SOH1 > SOH2. float.
        """
        z1, _, _ = self.model(x, key_padding_mask=key_padding_mask)
        z2, _, _ = self.model(x2, key_padding_mask=key_padding_mask2)

        if self.model.add_cls:
            z1_cls = z1[:, 0]
            z2_cls = z2[:, 0]
        else:
            raise NotImplementedError("SohCompareWrapper does not support add_cls=False")

        z_cat = torch.cat((z1_cls, z2_cls), dim=1)

        logits = self.regressor(z_cat).view(-1)  # (B,)

        target = target.to(dtype=logits.dtype, device=logits.device)

        loss = self.criterion(logits, target)

        return loss, logits
