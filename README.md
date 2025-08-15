# SOH-TEC

Implementation of **SOH-TEC**, a Transformer encoder for estimating EV battery state of health (SOH).
Includes a backbone encoder and task wrappers (regression, masked-token prediction, pairwise SOH comparison).

Paper: [Advancing state of health estimation for electric vehicles: Transformer-based approach leveraging real-world data](https://www.sciencedirect.com/science/article/pii/S266679242400026X)

---

## Install

Using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
uv pip install torch numpy
```

## Usage

### Initialize backbone
```python
import torch
from sohtec import SOHTECBackbone, SOHTECBackboneConfig, RegressionWrapper, MaskPredWrapper, SohCompareWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

B, C, L = 2, 4, 1800  # batch size, channels, sequence length

# SOHTEC backbone
cfg = SOHTECBackboneConfig(
    seq_len=L,   # seconds
    in_channels=C,  # [speed, voltage, current, soc]
)
backbone = SOHTECBackbone(cfg).to(device)
```

### Regression

```python
model_regression = RegressionWrapper(backbone, num_classes=1).to(device)
model_regression.eval()

x = torch.randn(B, C, L, device=device)  # (B,C,L)
key_padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)  # True=masked

with torch.inference_mode():
    y = model_regression(x, key_padding_mask)  # (B,1), predicted SOH
```

### Mask prediction

```python
model_mask_pred = MaskPredWrapper(backbone, mask_ratio=0.5).to(device)
model_mask_pred.eval()

with torch.inference_mode():
    loss = model_mask_pred(x, key_padding_mask)  # Reconstruction loss
```

### SOH comparison

```python
model_soh_compare = SohCompareWrapper(backbone).to(device)
model_soh_compare.eval()

x1 = torch.randn(B, C, L, device=device)
x2 = torch.randn(B, C, L, device=device)
key_padding_mask1 = torch.zeros(B, L, dtype=torch.bool, device=device)
key_padding_mask2 = torch.zeros(B, L, dtype=torch.bool, device=device)
target = torch.randint(0, 2, (B,), device=device)

with torch.inference_mode():
    loss, logits = model_soh_compare(x1, x2, target, key_padding_mask1, key_padding_mask2)
```

## Data & Shapes

- **Input tensor**: `(B, C, L)` = batch × channels × time
- **Channels**: depends on your data. Example: `[speed, voltage, current, soc]`
- **Sequence length `L`**: must equal `SOHTECBackboneConfig.seq_len`
- **Key padding mask**: shape `(B, L)`, `dtype=bool`, with **True = masked** (invalid/padded)

## License

MIT