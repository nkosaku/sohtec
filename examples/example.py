import torch

from sohtec import SOHTECBackbone, SOHTECBackboneConfig, RegressionWrapper, MaskPredWrapper, SohCompareWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

B, C, L = 2, 4, 1800  # batch size, channels, sequence length

# SOHTEC backbone
cfg = SOHTECBackboneConfig(
    seq_len=L,  # seconds
    in_channels=C,  # [speed, voltage, current, soc]
)
backbone = SOHTECBackbone(cfg).to(device)

# regression
model_regression = RegressionWrapper(backbone, num_classes=1).to(device)
model_regression.eval()

x = torch.randn(B, C, L, device=device)  # (B,C,L), dummy input
key_padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)  # True=masked

with torch.inference_mode():
    y = model_regression(x, key_padding_mask)  # (B,1), predicted value
print(y.shape, y)

# mask prediction
model_mask_pred = MaskPredWrapper(backbone, mask_ratio=0.5).to(device)
model_mask_pred.eval()

with torch.inference_mode():
    loss = model_mask_pred(x, key_padding_mask)  # Reconstruction loss
print(loss.item())

# soh comparison
model_soh_compare = SohCompareWrapper(backbone).to(device)
model_soh_compare.eval()

# dummy inputs and target
x1 = torch.randn(B, C, L, device=device)
x2 = torch.randn(B, C, L, device=device)
key_padding_mask1 = torch.zeros(B, L, dtype=torch.bool, device=device)
key_padding_mask2 = torch.zeros(B, L, dtype=torch.bool, device=device)
target = torch.randint(0, 2, (B,), device=device)  # 1 if SOH(x1) > SOH(x2), else 0

with torch.inference_mode():
    loss, logits = model_soh_compare(x1, x2, target, key_padding_mask1, key_padding_mask2)
print(loss.item())
print(logits.shape, logits)
