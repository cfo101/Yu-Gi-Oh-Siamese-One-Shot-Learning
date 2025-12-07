import os
import multiprocessing as mp
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import shutil

SOURCE_DIR = "yugioh_images"
OUT_DIR = "dataset_split"

TRAIN_DIR = os.path.join(OUT_DIR, "train")
VAL_DIR = os.path.join(OUT_DIR, "val")
TEST_DIR = os.path.join(OUT_DIR, "test")

AUGS_PER_VAL = 5  # moderate augmentations

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Mild augmentations for validation (not too heavy)
val_aug = T.Compose([
    T.ColorJitter(0.10, 0.10, 0.10, 0.05),
    T.RandomRotation(8),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
])

def process_one_file(filename):
    """
    Option A:
      - TEST = use only img1.jpg (real image)
      - VAL = augment img1.jpg a few times
      - TRAIN = all other real images (img2, img3, etc.)
    """

    src_path = os.path.join(SOURCE_DIR, filename)

    try:
        img = Image.open(src_path).convert("RGB")
    except:
        return f"Skipped (invalid): {filename}"

    # Extract the card ID and the view index
    # Example: 10000000_Obelisk_the_Tormentor_img1.jpg
    base, _ = os.path.splitext(filename)
    parts = base.split("_")

    if len(parts) < 3:
        return f"Bad name: {filename}"

    card_id = parts[0]
    view_tag = parts[-1]  # img1, img2, img3...

    # Case 1: TEST and VAL use img1
    if view_tag == "img1":
        # TEST gets the clean img1
        shutil.copyfile(src_path, os.path.join(TEST_DIR, filename))

        # VAL gets augmented versions of img1
        for i in range(AUGS_PER_VAL):
            aug = val_aug(img)
            out_name = f"{base}_val{i+1}.jpg"
            aug.save(os.path.join(VAL_DIR, out_name))

    # Case 2: TRAIN gets all real images except img1
    else:
        # Copy real image (img2, img3...)
        shutil.copyfile(src_path, os.path.join(TRAIN_DIR, filename))

    return f"Done: {filename}"


if __name__ == "__main__":
    files = [
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(files)} images.")
    print("Starting Option A split...\n")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_one_file, files), total=len(files)):
            pass

    print("\nDataset creation complete!")
    print(f"Train images: {len(os.listdir(TRAIN_DIR))}")
    print(f"Val images:   {len(os.listdir(VAL_DIR))}")
    print(f"Test images:  {len(os.listdir(TEST_DIR))}")
