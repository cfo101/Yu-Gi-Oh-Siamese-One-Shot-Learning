import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import open_clip
from paddleocr import PaddleOCR

# ===============================================================
# CONFIG
# ===============================================================
DATA_DIR = "dataset_split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")  # gallery
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

PROJ_PATH = "siamese_proj.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Fusion weights: alpha for visual, (1-alpha) for text
ALPHA_VISUAL = 0.7   # visual weight
ALPHA_TEXT = 0.3     # text weight

EVAL_VAL = True
EVAL_TEST = True  # set False if it’s too slow with OCR


# ===============================================================
# UTILITIES
# ===============================================================
def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None


def list_images(folder):
    """
    Returns list of dicts:
    {
        "path": full_path,
        "card_id": "10000000",
        "filename": "10000000_Obelisk_the_Tormentor_img1.jpg"
    }
    """
    out = []
    for f in os.listdir(folder):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        card_id = f.split("_")[0]
        full_path = os.path.join(folder, f)
        out.append(
            {"path": full_path, "card_id": card_id, "filename": f}
        )
    return out


def extract_card_name_from_filename(filename: str) -> str:
    """
    Example: '10000000_Obelisk_the_Tormentor_img1.jpg'
    → 'Obelisk the Tormentor'
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        name_parts = parts[1:-1]  # between ID and imgX
    elif len(parts) >= 2:
        name_parts = parts[1:]
    else:
        name_parts = parts

    name = " ".join(name_parts)
    # clean a bit
    name = name.replace("-", " ")
    return name


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
        meta = self.files[idx]
        path = meta["path"]
        card_id = meta["card_id"]
        filename = meta["filename"]

        img = load_image(path)
        if img is None:
            return None, card_id, path, filename

        return self.preprocess(img), card_id, path, filename


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    # -----------------------------------------------------------
    # Load CLIP model, tokenizer, preprocess
    # -----------------------------------------------------------
    print("Loading OpenCLIP ViT-B/16 encoder...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model = model.to(DEVICE)
    model.eval()
    preprocess = preprocess_val

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # -----------------------------------------------------------
    # Load projection head
    # -----------------------------------------------------------
    print("Loading trained projection head...")
    proj = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 128)
    ).to(DEVICE)
    proj.load_state_dict(torch.load(PROJ_PATH, map_location=DEVICE))
    proj.eval()

    # -----------------------------------------------------------
    # Initialize PaddleOCR (CPU by default)
    # -----------------------------------------------------------
    print("Initializing PaddleOCR (this may take a moment)...")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False
    )

    # -----------------------------------------------------------
    # Build GALLERY (visual embeddings with projection)
    # -----------------------------------------------------------
    print("Gallery images found:", len(os.listdir(TRAIN_DIR)))
    gallery_files = list_images(TRAIN_DIR)
    gallery_ids = []
    gallery_visual_embs = []

    print("\nBuilding gallery embeddings (projected)...")
    with torch.no_grad():
        for meta in tqdm(gallery_files, desc="Gallery-Proj"):
            img = load_image(meta["path"])
            if img is None:
                continue
            img_t = preprocess(img).unsqueeze(0).to(DEVICE)
            emb = model.encode_image(img_t)
            emb = proj(emb)
            emb = nn.functional.normalize(emb, dim=1)
            gallery_visual_embs.append(emb)
            gallery_ids.append(meta["card_id"])

    if len(gallery_visual_embs) == 0:
        raise RuntimeError("No valid gallery images found.")

    gallery_visual_embs = torch.cat(gallery_visual_embs, dim=0)  # [N, 128]
    print("Gallery built:", gallery_visual_embs.shape)

    # -----------------------------------------------------------
    # Build GALLERY TEXT embeddings (from filenames, one per ID)
    # -----------------------------------------------------------
    print("\nBuilding gallery TEXT embeddings (from filenames)...")

    # Map card_id -> canonical name string (from first filename)
    id_to_name = {}
    for meta in gallery_files:
        cid = meta["card_id"]
        if cid not in id_to_name:
            id_to_name[cid] = extract_card_name_from_filename(meta["filename"])

    unique_ids = list(id_to_name.keys())
    names = [id_to_name[cid] for cid in unique_ids]

    # Encode names with CLIP text encoder in batches
    id_to_text_emb = {}
    BATCH_TEXT = 256

    with torch.no_grad():
        for start in tqdm(range(0, len(unique_ids), BATCH_TEXT), desc="Gallery-Text"):
            batch_ids = unique_ids[start:start + BATCH_TEXT]
            batch_names = [id_to_name[cid] for cid in batch_ids]

            tokens = tokenizer(batch_names).to(DEVICE)
            text_emb = model.encode_text(tokens)
            text_emb = nn.functional.normalize(text_emb, dim=1)  # [B, d]

            for cid, emb_row in zip(batch_ids, text_emb):
                id_to_text_emb[cid] = emb_row.detach().clone()

    # Now build a [N_gallery, d] matrix aligned with gallery_ids
    text_dim = next(iter(id_to_text_emb.values())).shape[0]
    gallery_text_embs = torch.zeros(len(gallery_ids), text_dim, device=DEVICE)

    missing_ids = 0
    for i, cid in enumerate(gallery_ids):
        if cid in id_to_text_emb:
            gallery_text_embs[i] = id_to_text_emb[cid]
        else:
            # shouldn't happen, but just in case
            missing_ids += 1

    print(f"Text embeddings created for {len(id_to_text_emb)} unique cards. Missing IDs: {missing_ids}")

    # ===============================================================
    # OCR helper
    # ===============================================================
    def ocr_extract_text(path: str, fallback_name: str = "") -> str:
        """
        Run PaddleOCR on an image path and return a single text string.
        If OCR fails or finds nothing, fall back to filename-derived name.
        """
        try:
            result = ocr.ocr(path, cls=True)
        except Exception:
            return fallback_name

        if not result or len(result) == 0:
            return fallback_name

        # result is like: [ [ [box, (text, score)], ... ] ]
        texts = []
        for line in result:
            for det in line:
                txt, score = det[1][0], det[1][1]
                if score >= 0.5:
                    texts.append(txt)

        if not texts:
            return fallback_name

        return " ".join(texts)


    # ===============================================================
    # EVALUATION FUNCTION (with OCR fusion)
    # ===============================================================
    def evaluate_split(split_dir, title: str):
        print(f"\n====================================")
        print(f" Evaluating on: {title}")
        print(f"====================================")

        dataset = EvalSet(split_dir, preprocess)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

        top1 = top5 = 0
        total = 0

        with torch.no_grad():
            for imgs, ids, paths, filenames in tqdm(loader, desc=title):
                # Filter valid imgs
                good_imgs = []
                good_ids = []
                good_paths = []
                good_names = []

                for img, cid, path, fname in zip(imgs, ids, paths, filenames):
                    if img is not None:
                        good_imgs.append(img)
                        good_ids.append(cid)
                        good_paths.append(path)
                        # filename-derived name as fallback if OCR fails
                        good_names.append(extract_card_name_from_filename(fname))

                if len(good_imgs) == 0:
                    continue

                batch = torch.stack(good_imgs).to(DEVICE)

                # --------------------------------------------------
                # 1) Visual similarity (projection head)
                # --------------------------------------------------
                v = model.encode_image(batch)
                v = proj(v)
                v = nn.functional.normalize(v, dim=1)       # [B, 128]

                sims_visual = v @ gallery_visual_embs.T     # [B, N]

                # --------------------------------------------------
                # 2) Text similarity (PaddleOCR + CLIP text)
                # --------------------------------------------------
                # Extract OCR text for each query
                ocr_texts = []
                for path, fallback_name in zip(good_paths, good_names):
                    txt = ocr_extract_text(path, fallback_name)
                    if not txt.strip():
                        txt = fallback_name
                    ocr_texts.append(txt)

                # Encode OCR text with CLIP
                tokens = tokenizer(ocr_texts).to(DEVICE)
                t = model.encode_text(tokens)
                t = nn.functional.normalize(t, dim=1)       # [B, d_text]

                sims_text = t @ gallery_text_embs.T         # [B, N]

                # --------------------------------------------------
                # 3) Fuse visual + text similarities
                # --------------------------------------------------
                sims_fused = ALPHA_VISUAL * sims_visual + ALPHA_TEXT * sims_text

                topk = sims_fused.topk(5, dim=1).indices.cpu().numpy()

                for i, true_id in enumerate(good_ids):
                    preds = [gallery_ids[j] for j in topk[i]]

                    if preds[0] == true_id:
                        top1 += 1
                    if true_id in preds:
                        top5 += 1

                    total += 1

        print(f"\nResults for: {title}")
        print(f"Total queries: {total}")
        print(f"[Proj+OCR] Top-1: {top1} ({top1/total:.4f})")
        print(f"[Proj+OCR] Top-5: {top5} ({top5/total:.4f})")
        print("====================================\n")


    # ===============================================================
    # RUN EVAL
    # ===============================================================
    if EVAL_VAL:
        evaluate_split(VAL_DIR, "VAL")

    if EVAL_TEST:
        evaluate_split(TEST_DIR, "TEST")
