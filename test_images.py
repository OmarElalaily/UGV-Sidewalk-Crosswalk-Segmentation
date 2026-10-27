import torch
import segmentation_models_pytorch as smp
import cv2
import os
import numpy as np
from tqdm import tqdm
import os

''' ========================
    CONFIG
======================== '''
IMG_SIZE = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "output/test_moveable_aug_bright/test_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

''' ========================
    COLORS (BGR for OpenCV)
======================== '''
COLORS = {
    0: (0, 0, 255),    # crosswalk - blue
    1: (255, 0, 0),    # sidewalk + ped area - red
    2: (128, 128, 128),  # background - gray
}

''' ========================
    MODEL
======================== '''
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    classes=3,
    activation=None
)

model.load_state_dict(torch.load(
    "output/moveable_aug_bright/best_model.pth",
    map_location=device))
model.to(device)
model.eval()

# ========================
# Run on one image
# ========================
IMG_DIR  = "testing/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]

for img_name in images:
    IMG_PATH = os.path.join(IMG_DIR, img_name)
    
    image = cv2.imread(IMG_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = cv2.resize(image, (512, 512))

    inp = original.astype("float32") / 255.0
    inp = inp.transpose(2, 0, 1)
    inp = torch.tensor(inp).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(inp)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    colored = np.zeros((512, 512, 3), dtype=np.uint8)
    for class_id, color in COLORS.items():
        colored[pred == class_id] = color

    combined = np.hstack([original, colored])
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    
    save_path = os.path.join(OUTPUT_DIR, img_name.replace(".jpg", ".png"))
    cv2.imwrite(save_path, combined)
    print(f"Saved: {img_name}")

print("Done!")
