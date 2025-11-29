import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open_clip
from tqdm import tqdm

# ===============================================================
# CONFIG
# ===============================================================
DATA_DIR = "dataset_split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")  # gallery
VAL_DIR = os.path.join(DATA_DIR, "val")      # validation queries
TEST_DIR = os.path.join(DATA_DIR, "test")    # test queries

PROJ_PATH = "siamese_proj.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


# ===============================================================
# UTILITIES
# ===============================================================
def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None


def list_images(folder):
    out = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            card_id = f.split("_")[0]   # your filenames already use correct IDs
            out.append((os.path.join(folder, f), card_id))
    return out


# ===============================================================
# DATASET
# ===============================================================
class EvalSet(Dataset):
    def __init__(self, folder, preprocess):
        self.files = list_images(folder)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, card_id = self.files[idx]
        img = load_image(path)
        if img is None:
            return None, card_id
        return self.preprocess(img), card_id


# ===============================================================
# EVALUATION ON A FOLDER (val or test)
# ===============================================================
def evaluate(model, proj, preprocess, gallery_embs, gallery_ids, query_dir, title):
    print(f"\n====================================")
    print(f" Evaluating on: {title}")
    print(f"====================================")

    dataset = EvalSet(query_dir, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2)

    top1 = top5 = 0
    top1_clip = top5_clip = 0
    total = 0

    with torch.no_grad():
        for imgs, ids in tqdm(loader, desc=title):

            # Filter out corrupt images
            good_imgs = []
            good_ids = []
            for img, cid in zip(imgs, ids):
                if img is not None:
                    good_imgs.append(img)
                    good_ids.append(cid)

            if len(good_imgs) == 0:
                continue

            batch = torch.stack(good_imgs).to(DEVICE)

            # -----------------------------
            # Projected embeddings
            # -----------------------------
            q = model.encode_image(batch)
            q = proj(q)
            q = nn.functional.normalize(q, dim=1)

            sims = q @ gallery_embs.T
            topk = sims.topk(5, dim=1).indices.cpu().numpy()

            # -----------------------------
            # CLIP-only baseline
            # -----------------------------
            q_clip = model.encode_image(batch)
            q_clip = nn.functional.normalize(q_clip, dim=1)

            sims_clip = q_clip @ gallery_embs_clip.T
            topk_clip = sims_clip.topk(5, dim=1).indices.cpu().numpy()

            # -----------------------------
            # Count accuracy
            # -----------------------------
            for i, true_id in enumerate(good_ids):
                pred_proj = [gallery_ids[j] for j in topk[i]]
                pred_clip = [gallery_ids[j] for j in topk_clip[i]]

                # projected
                if pred_proj[0] == true_id:
                    top1 += 1
                if true_id in pred_proj:
                    top5 += 1

                # CLIP
                if pred_clip[0] == true_id:
                    top1_clip += 1
                if true_id in pred_clip:
                    top5_clip += 1

                total += 1

    # -----------------------------
    # Print results
    # -----------------------------
    print(f"\nResults for: {title}")
    print(f"Total queries: {total}")
    print(f"[Proj] Top-1: {top1} ({top1/total:.4f})")
    print(f"[Proj] Top-5: {top5} ({top5/total:.4f})")
    print(f"[CLIP] Top-1: {top1_clip} ({top1_clip/total:.4f})")
    print(f"[CLIP] Top-5: {top5_clip} ({top5_clip/total:.4f})")
    print("====================================\n")


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    print("Loading OpenCLIP ViT-B/16 encoder...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model = model.to(DEVICE)
    model.eval()
    preprocess = preprocess_val

    print("Loading trained projection head...")
    proj = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 128)
    ).to(DEVICE)
    proj.load_state_dict(torch.load(PROJ_PATH, map_location=DEVICE))
    proj.eval()

    # ===============================================================
    # BUILD GALLERY (train/)
    # ===============================================================
    print("\nBuilding gallery embeddings...")
    gallery_files = list_images(TRAIN_DIR)
    gallery_ids = []
    gallery_embs = []

    with torch.no_grad():
        for path, cid in tqdm(gallery_files, desc="Gallery"):
            img = load_image(path)
            if img is None:
                continue
            img_t = preprocess(img).unsqueeze(0).to(DEVICE)

            emb = model.encode_image(img_t)
            emb = proj(emb)
            emb = nn.functional.normalize(emb, dim=1)

            gallery_embs.append(emb)
            gallery_ids.append(cid)

    gallery_embs = torch.cat(gallery_embs, dim=0)

    print(f"Gallery built: {len(gallery_embs)} images.")

    # ===============================================================
    # BUILD GALLERY (CLIP-only baseline)
    # ===============================================================
    print("Building CLIP-only gallery...")
    gallery_embs_clip = []
    with torch.no_grad():
        for path, cid in tqdm(gallery_files, desc="Gallery-CLIP"):
            img = load_image(path)
            if img is None:
                continue
            img_t = preprocess(img).unsqueeze(0).to(DEVICE)
            emb = nn.functional.normalize(model.encode_image(img_t), dim=1)
            gallery_embs_clip.append(emb)

    gallery_embs_clip = torch.cat(gallery_embs_clip, dim=0)


    # ===============================================================
    # EVAL ON VAL + TEST SETS
    # ===============================================================
    evaluate(model, proj, preprocess, gallery_embs, gallery_ids, VAL_DIR, title="VAL")
    evaluate(model, proj, preprocess, gallery_embs, gallery_ids, TEST_DIR, title="TEST")
