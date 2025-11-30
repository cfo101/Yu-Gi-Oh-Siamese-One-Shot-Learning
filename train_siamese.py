from __future__ import annotations
import os
import random
from typing import Optional
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open_clip
from tqdm import tqdm

# ===============================================================
# CONFIG
# ===============================================================
DATA_DIR = "dataset_split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4


# ===============================================================
# TRAINING DATA AUGMENTATION (mild → stable)
# ===============================================================
augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(0.10, 0.10, 0.10, 0.05),
])


# ===============================================================
# CLIP PREPROCESS
# ===============================================================
def load_clip_transform():
    _, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    return preprocess

clip_preprocess = load_clip_transform()


# ===============================================================
# DATASET
# ===============================================================
class SiameseDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.files = []
        self.by_card = {}

        for f in os.listdir(root_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            card_id = f.split("_")[0]

            self.files.append(f)
            if card_id not in self.by_card:
                self.by_card[card_id] = []
            self.by_card[card_id].append(f)

        self.card_ids = list(self.by_card.keys())

    def __len__(self):
        return len(self.files)

    def load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except:
            return None

    def synthetic_positive(self, img):
        return augment(img)

    def __getitem__(self, idx):
        anchor_filename = self.files[idx]
        anchor_id = anchor_filename.split("_")[0]
        anchor_path = os.path.join(self.root, anchor_filename)

        anchor_img = self.load_image(anchor_path)
        if anchor_img is None:
            return self.__getitem__(random.randint(0, len(self)-1))

        # ---------------------------
        # POSITIVE
        # ---------------------------
        same_card_images = self.by_card[anchor_id]

        if len(same_card_images) > 1:
            pos_filename = random.choice([f for f in same_card_images if f != anchor_filename])
            pos_img = self.load_image(os.path.join(self.root, pos_filename))
            if pos_img is None:
                pos_img = self.synthetic_positive(anchor_img)
        else:
            pos_img = self.synthetic_positive(anchor_img)

        # ---------------------------
        # NEGATIVE (ignored for batch-hard)
        # ---------------------------
        # We still sample one, but we won't use it directly
        neg_id = random.choice([cid for cid in self.card_ids if cid != anchor_id])
        neg_filename = random.choice(self.by_card[neg_id])
        neg_img = self.load_image(os.path.join(self.root, neg_filename))
        if neg_img is None:
            return self.__getitem__(random.randint(0, len(self)-1))

        # ---------------------------
        # Preprocess for CLIP
        # ---------------------------
        anchor_img = clip_preprocess(anchor_img)
        pos_img = clip_preprocess(pos_img)
        neg_img = clip_preprocess(neg_img)   # still returned (unused directly)

        return anchor_img, pos_img, neg_img


# ===============================================================
# MODEL
# ===============================================================
def build_model():
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model = model.to(DEVICE)
    model.eval()

    # Freeze encoder
    for p in model.parameters():
        p.requires_grad = False

    # Trainable projection head
    proj = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 128)
    ).to(DEVICE)

    return model, proj


# ===============================================================
# TRAINING LOOP (Batch-Hard Negative Mining)
# ===============================================================
def train_siamese():
    print("Loading CLIP model...")
    model, proj = build_model()

    dataset = SiameseDataset(TRAIN_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(proj.parameters(), lr=LR)
    triplet_loss = nn.TripletMarginLoss(margin=0.3)

    print("Starting training...")
    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0

        for anchor_img, pos_img, _ in loop:
            anchor_img = anchor_img.to(DEVICE, non_blocking=True)
            pos_img = pos_img.to(DEVICE, non_blocking=True)

            # ---------------------------
            # Encode with CLIP
            # ---------------------------
            with torch.no_grad():
                a = model.encode_image(anchor_img)
                p = model.encode_image(pos_img)

            # Project + normalize
            a = nn.functional.normalize(proj(a), dim=1)
            p = nn.functional.normalize(proj(p), dim=1)

            # ---------------------------
            # BATCH-HARD NEGATIVE MINING
            # ---------------------------
            # Pairwise anchor distances (L2)
            dist_matrix = torch.cdist(a, a, p=2)

            # Mask out diagonal (self-distances)
            mask = ~torch.eye(a.size(0), dtype=bool, device=DEVICE)

            # Hardest negative index per anchor = closest other sample
            hard_neg_idx = dist_matrix[mask].reshape(a.size(0), -1).argmin(dim=1)

            # Hard negatives
            n = a[hard_neg_idx]

            # ---------------------------
            # Triplet loss
            # ---------------------------
            loss = triplet_loss(a, p, n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n + 1e-8))

    print("Training complete!")
    torch.save(proj.state_dict(), "siamese_proj.pth")
    print("Projection head saved.")


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    train_siamese()
