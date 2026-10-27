import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp

IMGSZ = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "videos/input.mp4"
OUTPUT_DIR = "output/test_video"
os.makedirs(OUTPUT_DIR, exist_ok=True)

''' ========================
    COLORS (BGR for OpenCV)
======================== '''
COLORS = {
    0: (255, 0, 0),      # crosswalk - blue (BGR)
    1: (0, 0, 255),      # sidewalk - red (BGR)
    # 2: (0, 255, 255),      # grass - yellow (BGR)  
    2: (128, 128, 128),  # background - gray (BGR)
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

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "output_video3_aug_bright.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (IMGSZ * 2, IMGSZ)  # side by side
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = cv2.resize(frame, (IMGSZ, IMGSZ))
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    inp = rgb.astype("float32") / 255.0
    inp = inp.transpose(2, 0, 1)
    inp = torch.tensor(inp).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(inp)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    colored = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    for class_id, color in COLORS.items():
        colored[pred == class_id] = color

    combined = np.hstack([original, colored])
    out.write(combined)

cap.release()
out.release()
print(f"Done! Video saved to: {OUTPUT_DIR}")
