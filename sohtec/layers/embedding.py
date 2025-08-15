import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        no_activation=False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=no_activation,
        )

        # for mask
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=False)

        self.no_activation = no_activation
        if not no_activation:
            self.norm = MaskedBatchNorm1d(out_channels)
            self.act = nn.ReLU()

        # weight initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def out_len(self, in_len):
        return (in_len - self.conv.kernel_size[0] + 2 * self.conv.padding[0]) // self.conv.stride[0] + 1

    def forward(self, x, mask=None):
        assert x.dim() == 3, f"x dim must be 3, got {x.dim()}"
        B, C, L = x.shape

        x = self.conv(x)

        if mask is not None:
            assert mask.dtype == torch.bool, f"mask must be bool, got {mask.dtype}"
            assert mask.shape == (B, L), f"mask shape must be (B, L), got {mask.shape}"
            with torch.no_grad():
                mask = mask.unsqueeze(1)
                mask_float = mask.float()
                mask = self.max_pool(mask_float)
                mask = mask.bool().squeeze(1)

        if not self.no_activation:
            x = self.norm(x, mask)
            x = self.act(x)
        return x, mask


class MultiLayerPatchEmbed(nn.Module):
    def __init__(
        self,
        seq_len=1800,
        in_channels=4,
        embed_dim=256,
        k1=4,
        s1=2,
        k=3,
        s=2,
        l=2,  # noqa: E741
    ):
        super().__init__()
        self.seq_len = seq_len

        self.blocks = nn.ModuleList([])
        for i in range(l):
            final_layer = i == l - 1
            if i == 0:
                self.blocks.append(
                    CNNBlock(
                        in_channels,
                        embed_dim,
                        k1,
                        s1,
                        no_activation=final_layer,
                    )
                )
            else:
                self.blocks.append(CNNBlock(embed_dim, embed_dim, k, s, no_activation=final_layer))

    @property
    def num_patches(self):
        out_len = self.seq_len
        for block in self.blocks:
            out_len = block.out_len(out_len)
        return out_len

    def forward(self, x, mask=None):
        """x: (N, C, L) -> (N, L', embed_dim)"""
        assert x.shape[2] == self.seq_len, f"expected L={self.seq_len}, got {x.shape[2]}"

        for block in self.blocks:
            x, mask = block(x, mask)
        x = x.transpose(1, 2)
        return x, mask


class MaskedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, x, mask=None):
        """
        x: (B, C, T)
        mask: (B, T). True for mask, False for non-mask
        """
        B, C, T = x.shape
        assert C == self.num_features

        x32 = x.float()

        if mask is None:
            mean = x32.mean(dim=(0, 2))
            var = x32.var(dim=(0, 2), unbiased=False)
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                    self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
                    self.num_batches_tracked.add_(1)
            if not self.training and self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            x_hat = (x32 - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
            y = x_hat
            if self.affine:
                y = y * self.weight[None, :, None] + self.bias[None, :, None]
            return y.type_as(x)

        mask = mask.to(torch.bool)
        w = (~mask).to(x.dtype).unsqueeze(1)
        w32 = w.float()

        count = w32.sum(dim=(0, 2))
        if torch.any(count == 0):
            raise ValueError("All timesteps in this batch are masked.")
        else:
            sum_ = (x32 * w32).sum(dim=(0, 2))  # (C,)
            mean = sum_ / count  # (C,)

            diff = x32 - mean.unsqueeze(-1)
            var_num = (diff * diff * w32).sum(dim=(0, 2))  # (C,)
            var = var_num / count

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
                self.num_batches_tracked.add_(1)

        if not self.training and self.track_running_stats:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x32 - mean.unsqueeze(-1)) / torch.sqrt(var.unsqueeze(-1) + self.eps)

        # zero out invalid positions
        y = x_hat * w32

        if self.affine:
            y = y * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1) * w32

        return y.type_as(x)
