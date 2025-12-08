import os
import multiprocessing as mp
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import shutil

# ===========================================================
# CONFIG
# ===========================================================
SOURCE_DIR = "yugioh_images"
OUT_DIR = "dataset_split"

TRAIN_DIR = os.path.join(OUT_DIR, "train")
VAL_DIR = os.path.join(OUT_DIR, "val")
TEST_DIR = os.path.join(OUT_DIR, "test")

AUGS_PER_VAL = 5
AUGS_PER_TEST = 5

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# ===========================================================
# TRANSFORMS
# ===========================================================
val_aug = T.Compose([
    T.ColorJitter(0.15, 0.15, 0.15, 0.05),
    T.RandomRotation(10),
    T.RandomResizedCrop(224, scale=(0.75, 1.0)),
])
test_aug = T.Compose([
    T.ColorJitter(0.25, 0.25, 0.25, 0.15),
    T.RandomRotation(15),
    T.RandomPerspective(distortion_scale=0.35, p=0.7),
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
])
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

# ===========================================================
# WORKER FUNCTION (runs in parallel)
# ===========================================================
def process_one_file(filename):
    src_path = os.path.join(SOURCE_DIR, filename)

    try:
        img = Image.open(src_path).convert("RGB")
    except:
        return f"Skipped (invalid): {filename}"

    base = os.path.splitext(filename)[0]

    # Copy original to train/ (no augment)
    shutil.copyfile(src_path, os.path.join(TRAIN_DIR, filename))

    # Generate VAL augmentations
    for i in range(AUGS_PER_VAL):
        aug = val_aug(img)
        out_name = f"{base}_val{i+1}.jpg"
        out_path = os.path.join(VAL_DIR, out_name)
        aug.save(out_path)

    # Generate TEST augmentations
    for i in range(AUGS_PER_TEST):
        aug = test_aug(img)
        out_name = f"{base}_test{i+1}.jpg"
        out_path = os.path.join(TEST_DIR, out_name)
        aug.save(out_path)

    return f"Done: {filename}"


# ===========================================================
# MAIN (Parallel)
# ===========================================================
if __name__ == "__main__":
    files = [
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(files)} images.")
    print(f"Using {mp.cpu_count()} CPU cores.")
    print("Starting parallel augmentation...\n")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_one_file, files), total=len(files)):
            pass

    print("\nDataset creation complete!")
    print(f"Train images: {len(os.listdir(TRAIN_DIR))}")
    print(f"Val images:   {len(os.listdir(VAL_DIR))}")
    print(f"Test images:  {len(os.listdir(TEST_DIR))}")