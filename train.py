import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import csv
import numpy as np
from tqdm import tqdm
import albumentations as A
import matplotlib
matplotlib.use("Agg")  # no display needed on the training machine
import matplotlib.pyplot as plt

OUTPUT_DIR = "output/moveable_aug_bright"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ========================
# CONFIG
# ========================
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# CLASS MAPPING
# ========================
COLOR_TO_CLASS = {
    (140, 140, 200): 0,   # crosswalk
    (244,  35, 232): 1,   # sidewalk
    ( 96,  96,  96): 1,   # pedestrian area -> same as sidewalk
    # everything else -> 2 (background) via np.full default
}

# NUM_CLASSES = len(set(COLOR_TO_CLASS.values()))
NUM_CLASSES = 3  # crosswalk, sidewalk, background
CLASS_NAMES = ["Crosswalk", "Sidewalk", "Background"]
BACKGROUND_ID = 2  # valid labels are 0,1,2 for a 3-class model

# ========================
# AUGMENTATIONS
# ========================
transform = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10),
             cval_mask=BACKGROUND_ID, p=0.3),  # fill rotated borders with background (newer: fill_mask)
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02, p=0.5),
])

# ========================
# DATASET
# ========================
class MapillaryDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.images = []

        for f in os.listdir(img_dir):
            if not f.endswith(".jpg"):
                continue

            mask_path = os.path.join(mask_dir, f.replace(".jpg", ".png"))

            if os.path.exists(mask_path):
                self.images.append(f)

        print(f"Total valid images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.img_dir,  self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        new_mask = np.full((mask.shape[0], mask.shape[1]), BACKGROUND_ID, dtype=np.uint8)  # default = background (2)
        for (r, g, b), class_id in COLOR_TO_CLASS.items():
            match = (mask[:, :, 0] == r) & (mask[:, :, 1] == g) & (mask[:, :, 2] == b)
            new_mask[match] = class_id

        # augmentation only for training
        if self.augment:
            augmented = transform(image=image, mask=new_mask)
            image    = augmented["image"]
            new_mask = augmented["mask"]

        image = image.astype("float32") / 255.0
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), torch.tensor(new_mask, dtype=torch.long)

# ========================
# LOAD DATA
# ========================
train_dataset = MapillaryDataset("training/images", "training/v2.0/labels", augment=True)
val_dataset   = MapillaryDataset("validation/images", "validation/v2.0/labels", augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

# check multiple images
for i in range(10):
    img, mask = train_dataset[i]
    print(f"Image {i}: {torch.unique(mask).tolist()}")
# ========================
# MODEL
# ========================
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    classes=NUM_CLASSES,
    activation=None

).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
class_weights = torch.tensor([3.0, 1.5, 0.5]).to(device)  # crosswalk, sidewalk, background
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=5,
    factor=0.5)

# ========================
# GRAPH / METRIC SAVING
# ========================
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "accuracy": [],
    "miou": [],
    "iou_per_class": [],  # list of arrays, one per epoch
}


def save_training_graphs(history, out_dir):
    epochs = history["epoch"]
    if len(epochs) == 0:
        return
    iou_arr = np.array(history["iou_per_class"])  # shape (num_epochs, NUM_CLASSES)

    # 1. Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # 2. mIoU curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["miou"], label="mIoU", color="green", marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Mean IoU over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "miou_curve.png"), dpi=300)
    plt.close()

    # 3. Per-class IoU curve
    plt.figure(figsize=(8, 5))
    for c in range(iou_arr.shape[1]):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class {c}"
        plt.plot(epochs, iou_arr[:, c], label=name, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Per-Class IoU over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_iou.png"), dpi=300)
    plt.close()

    # 4. Pixel accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="Pixel Accuracy", color="purple", marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Pixel Accuracy over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    # CSV of all metrics (so you can re-plot later without retraining)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "train_loss", "val_loss", "accuracy", "miou"] + \
                 [f"iou_{CLASS_NAMES[c].lower()}" if c < len(CLASS_NAMES) else f"iou_class_{c}"
                  for c in range(iou_arr.shape[1])]
        writer.writerow(header)
        for i in range(len(epochs)):
            writer.writerow([epochs[i], history["train_loss"][i], history["val_loss"][i],
                             history["accuracy"][i], history["miou"][i]] + list(iou_arr[i]))


# ========================
# TRAINING LOOP
# ========================
best_miou = 0

for epoch in range(EPOCHS):
    train_loss = 0
    '''===== TRAIN ===== '''
    model.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    intersection = torch.zeros(NUM_CLASSES).to(device)
    union = torch.zeros(NUM_CLASSES).to(device)

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == masks).sum().item()
            total += masks.numel()

            for cls in range(NUM_CLASSES):
                pred_cls = (preds == cls)
                mask_cls = (masks == cls)

                intersection[cls] += (pred_cls & mask_cls).sum().item()
                union[cls] += (pred_cls | mask_cls).sum().item()

    accuracy = correct / total
    iou = intersection / (union + 1e-6)
    miou = iou.mean()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"mIoU: {miou:.4f}")
    scheduler.step(miou)
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
    # torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pth"))

    # ----- record metrics for graphs -----
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["accuracy"].append(accuracy)
    history["miou"].append(miou.item())
    history["iou_per_class"].append(iou.detach().cpu().numpy())

    # regenerate graphs every epoch so a crash/early-stop still leaves the latest figures
    save_training_graphs(history, OUTPUT_DIR)

    # save best model
    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        print("Best model saved!")
    with open(os.path.join(OUTPUT_DIR, "log.txt"), "a") as f:
        f.write(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, mIoU: {miou:.4f}\n")
# ========================
# SAVE MODEL
# ========================
torch.save(model.state_dict(), "model.pth")
print("Model saved")

# final graph save (in case loop finished cleanly)
save_training_graphs(history, OUTPUT_DIR)
print(f"Training graphs saved to {OUTPUT_DIR}")