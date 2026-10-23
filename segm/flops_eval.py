import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from segm.model.factory import load_model, create_segmenter

# =========================
# CONFIG
# =========================
checkpoint_path = "/ivi/xfs/skalra/seg_shard_poly/checkpoint.pth"
config_path = "/ivi/xfs/skalra/seg_shard_poly/variant.yml"
image_size = 512   # ADE20K standard
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = {
    'backbone':"deit_tiny_patch16_224",
    "n_cls": 150,   # ADE20K classes

    # =========================
    # ENCODER
    # =========================
    "image_size": (512,512),
    "patch_size": 16,

    "d_model": 192,
    "n_heads": 3,
    "n_layers": 12,
    
    "normalization": "deit",
    "distilled": False,

    # =========================
    # DECODER
    # =========================
    "decoder": {"name": "linear"}}

# =========================
# LOAD MODEL
# =========================
# model, _ = load_model(checkpoint_path)
# checkpoint = torch.load(checkpoint_path, map_location="cpu")

# load weights (ignore mismatches like pos_embed if needed)
# model.load_state_dict(checkpoint["model"], strict=True)

model = create_segmenter(cfg)

model.to(device)
model.eval()

print(parameter_count_table(model))

# =========================
# DUMMY INPUT
# =========================
dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

flops = FlopCountAnalysis(model, dummy_input)
total_flops = flops.total()

print(f"FLOPs: {total_flops / 1e9:.2f} GFLOPs")
exit()
# =========================
# FLEX LEVEL FLOPs
# =========================
max_level = model.encoder.max_level()

print("\n==== FLOPs per level ====\n")

for level in range(max_level + 1):
    print(f"\n--- Level {level} ---")

    # set level
    model.encoder.set_level_use(level)

    # wrap forward because your model expects level arg
    def forward_func(x):
        return model(x, level=level)

    # flops = FlopCountAnalysis(model, dummy_input)
    flops = FlopCountAnalysis(model, (dummy_input, level))
    total_flops = flops.total()

    print(f"FLOPs: {total_flops / 1e9:.2f} GFLOPs")