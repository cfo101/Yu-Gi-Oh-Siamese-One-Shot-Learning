import os
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# ======================================================
# CONFIG
# ======================================================
DATA_DIR = "dataset_split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
AUGS_PER_IMAGE = 10

# Create val directory if missing
os.makedirs(VAL_DIR, exist_ok=True)

# ======================================================
# AUGMENTATION PIPELINE
# ======================================================
augment = T.Compose([
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.RandomRotation(10),
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomPerspective(distortion_scale=0.25, p=0.5),
    T.RandomApply([T.GaussianBlur(5)], p=0.3),
])

# Needed before saving back to image
to_pil = T.ToPILImage()
to_tensor = T.ToTensor()

# ======================================================
# MAIN
# ======================================================
files = [f for f in os.listdir(TRAIN_DIR)
         if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"Found {len(files)} training images.")
print(f"Generating {AUGS_PER_IMAGE} augmentations per image...")
print(f"Total val images = {len(files) * AUGS_PER_IMAGE}")

for fname in tqdm(files):
    card_id = fname.split("_")[0]  # keep whatever your filenames have
    base = os.path.splitext(fname)[0]

    img_path = os.path.join(TRAIN_DIR, fname)

    try:
        img = Image.open(img_path).convert("RGB")
    except:
        continue

    for i in range(AUGS_PER_IMAGE):
        # Apply augmentation
        aug = augment(img)

        # Save
        out_name = f"{base}_aug{i+1}.jpg"
        out_path = os.path.join(VAL_DIR, out_name)
        aug.save(out_path)

print("Done! Validation set created successfully.")
