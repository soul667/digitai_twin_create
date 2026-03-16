import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")
example_path = "da3/frames"

# Recursively read frames because extractor saves to da3/frames/<video_name>/*.{png,jpg,jpeg}.
images = []
for ext in ("*.png", "*.jpg", "*.jpeg"):
    images.extend(glob.glob(os.path.join(example_path, "**", ext), recursive=True))
images = sorted(images)
if len(images) == 0:
    raise FileNotFoundError(
        f"No images found under '{example_path}'. "
        "Expected files like da3/frames/<video_name>/*.(png|jpg|jpeg)"
    )

# Avoid feeding too many views at once; DA3 expects multi-view inputs and memory grows with N.
max_images = int(os.environ.get("MAX_IMAGES", "12"))
images = images[:max_images]
print(f"Using {len(images)} image(s) for inference")

infer_gs = os.environ.get("INFER_GS", "0") == "1"
export_dir = os.environ.get("EXPORT_DIR", "").strip() or None
export_format = os.environ.get("EXPORT_FORMAT", "mini_npz")

print(f"infer_gs={infer_gs}, export_format={export_format}, export_dir={export_dir}")
prediction = model.inference(
    images,
    infer_gs=infer_gs,
    export_dir=export_dir,
    export_format=export_format,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)

if export_dir is not None:
    print(f"Exported results to: {export_dir}")
